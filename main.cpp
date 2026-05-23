/* PowerSpectrumExtractor
 * Reads one or more AMReX plot files, extracts a named real-space component
 * (default "R"), forward-FFTs it, applies the same physical normalisation used
 * in RandomField::extract(), and bins the power spectrum exactly as
 * RandomField::print_power_spectrum() does.
 *
 * Inputs (via ParmParse / AMReX inputs file or command-line key=value pairs):
 *   plotfiles   = plt00000 plt00100 ...   (explicit list of plot-file directories)
 *   plotfile_dir = /path/to/run/          (scan this directory for all plot files)
 *   output_dir  = spectra/                (directory to write spectrum .dat files)
 *   component   = R                       (component name; default "R")
 *   L           = 1.0                     (physical box length; default: read from file)
 *
 * plotfiles and plotfile_dir may both be specified; results are merged and sorted.
 *
 * Output:  one file per plot file, named  <output_dir>/spectrum-<basename>.dat
 *          Two-column ASCII: k_mid  P(k)
 *
 * Build:  see GNUmakefile in this directory.
 */

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_FFT.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuAtomic.H>

#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <sys/stat.h>

using namespace amrex;

// ---------------------------------------------------------------------------
// Helpers that replicate the private RandomField methods used inside
// print_power_spectrum.
// ---------------------------------------------------------------------------

// Replicates RandomField::invert_index
// Maps a j or k index into the Hermitian-symmetric half [0, N/2].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int invert_index(int indx, int N)
{
    return static_cast<int>(N/2 - std::abs(N/2 - indx));
}

// Replicates RandomField::get_kmag
// Returns the physical wavenumber magnitude at Fourier-space index iv.
// The first index i is already in [0, N/2] (FFTW R2C convention).
// The j and k indices wrap around via invert_index.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real get_kmag(IntVect iv, int N, Real L)
{
    const int i = iv[0];
    const int j = invert_index(iv[1], N);
    const int k = invert_index(iv[2], N);
    return std::sqrt(Real(i*i + j*j + k*k)) * 2.0 * M_PI / L;
}

// ---------------------------------------------------------------------------
// Power spectrum function — replicates RandomField::print_power_spectrum
// exactly, except that it writes to a plain text file instead of SmallDataIO.
//
// field_k  : Fourier-space cMultiFab, already normalised by 1/(norm*N^3)
//            (same state as R_k just before the call in RandomField::extract)
// N        : cubic grid size
// L        : physical box length
// output_path : path of the output .dat file
// component   : which component of field_k to use (default 0)
// ---------------------------------------------------------------------------
void compute_power_spectrum(cMultiFab       &field_k,
                            int              N,
                            Real             L,
                            const std::string &output_path,
                            int              component = 0)
{
    // --- Set up the isotropic k axis (identical to print_power_spectrum) ---
    const Real kiso_max = std::sqrt(3.0) * N * M_PI / L;
    const Real dkiso    = std::sqrt(3.0) * 2.0 * M_PI / L;
    constexpr Real tol  = 1.0e-12;

    if (kiso_max / dkiso - N/2 > tol)
        Error("compute_power_spectrum: Isotropic k axis check failed.");

    // kiso has N/2+1 entries (index 0 … N/2), ps_map and kcount likewise.
    Vector<Real> kiso  (N/2 + 1, 0.0);
    Vector<Real> ps_map(N/2 + 1, 0.0);
    Vector<int>  kcount(N/2 + 1, 0);

    for (int s = 0; s <= N/2; ++s) { kiso[s] = s * dkiso; }

    // --- Bin the power at each Fourier mode ---
    // This loop mirrors the ParallelFor in print_power_spectrum verbatim,
    // including the Hermitian factor-of-2 for modes with 0 < i < N/2.
    MFIter::allowMultipleMFIters(true);
    for (MFIter mfi(field_k); mfi.isValid(); ++mfi)
    {
        auto const& fp  = field_k.array(mfi);
        const Box&  bx  = mfi.fabbox();

        ParallelFor(bx, [=, &ps_map, &kcount]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            IntVect iv{i, j, k};
            Real kmag = get_kmag(iv, N, L);

            if (kmag - kiso_max > tol)
            {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                    "compute_power_spectrum: kmag > kiso_max");
            }

            for (int s = 1; s <= N/2; ++s)
            {
                if (kmag < kiso[0])
                {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                        "compute_power_spectrum: kmag below kiso domain");
                }
                else if (kmag - kiso[N/2] > tol)
                {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                        "compute_power_spectrum: kmag above kiso domain");
                }
                else if (kmag < kiso[s] && kmag >= kiso[s-1])
                {
                    Real power = (fp(i,j,k,component).real() * fp(i,j,k,component).real()
                                + fp(i,j,k,component).imag() * fp(i,j,k,component).imag());
                    if (i != 0 && i != N/2) { power *= 2.0; }
                    Gpu::Atomic::Add(&kcount[s-1], 1);
                    Gpu::Atomic::Add(&ps_map[s-1], power);
                    break;
                }
                else if (kmag == kiso[N/2])
                {
                    Real power = (fp(i,j,k,component).real() * fp(i,j,k,component).real()
                                + fp(i,j,k,component).imag() * fp(i,j,k,component).imag());
                    if (i != 0 && i != N/2) { power *= 2.0; }
                    Gpu::Atomic::Add(&kcount[N/2], 1);
                    Gpu::Atomic::Add(&ps_map[N/2], power);
                    break;
                }
                else { continue; }
            }
        });
    }

    // Reduce across MPI ranks
    ParallelAllReduce::Sum(kcount.data(), static_cast<int>(kcount.size()),
                           ParallelContext::CommunicatorSub());
    ParallelAllReduce::Sum(ps_map.data(), static_cast<int>(ps_map.size()),
                           ParallelContext::CommunicatorSub());

    // --- Write output (IO processor only) ---
    // Matches the SmallDataIO::write_data_line loop in print_power_spectrum.
    if (ParallelDescriptor::IOProcessor())
    {
        std::ofstream ofs(output_path);
        if (!ofs)
            Abort("compute_power_spectrum: cannot open " + output_path);

        ofs << std::scientific;
        ofs.precision(14);

        for (int s = 0; s < N/2; ++s)
        {
            Real k_mid  = (kiso[s] + kiso[s+1]) * 0.5;
            Real ps_val = (kcount[s] > 0) ? ps_map[s] / Real(kcount[s]) : 0.0;
            ofs << k_mid << "  " << ps_val << "\n";
        }
    }

    Print() << "  Written: " << output_path << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Initialize(argc, argv);
    {
        // ---- Parse inputs ----
        ParmParse pp;

        // Build the list of plot-file directories to process.
        // Accepts an explicit list, a directory to scan, or both.
        Vector<std::string> plotfiles;

        // Optional: explicit list of plot-file directories
        int n_pf = pp.countval("plotfiles");
        if (n_pf > 0)
        {
            plotfiles.resize(n_pf);
            pp.getarr("plotfiles", plotfiles, 0, n_pf);
        }

        // Optional: directory containing plot files — scan for all subdirs
        // that contain an AMReX "Header" file.
        std::string plotfile_dir;
        if (pp.query("plotfile_dir", plotfile_dir))
        {
            namespace fs = std::filesystem;
            if (!fs::is_directory(plotfile_dir))
                Abort("plotfile_dir is not a directory: " + plotfile_dir);

            std::vector<std::string> found;
            for (const auto& entry : fs::directory_iterator(plotfile_dir))
            {
                if (entry.is_directory())
                {
                    // An AMReX plot file directory always contains a "Header" file.
                    fs::path header = entry.path() / "Header";
                    if (fs::exists(header))
                        found.push_back(entry.path().string());
                }
            }
            std::sort(found.begin(), found.end());
            for (const auto& p : found) plotfiles.push_back(p);
        }

        if (plotfiles.empty())
            Abort("No plot files found. Specify plotfiles=... or plotfile_dir=...");
        else
            Print() << "Found " << plotfiles.size() << " files at " << plotfile_dir << "\n";

        // Required: output directory
        std::string output_dir;
        pp.get("output_dir", output_dir);
        // Strip trailing slash for consistency
        while (output_dir.size() > 1 && output_dir.back() == '/') output_dir.pop_back();

        // Optional: component name (default "R")
        std::string comp_name = "";
        pp.query("component", comp_name);
        if (comp_name == "")
            Abort("No component found. Please specify a MultiFab component in the params file.");

        // Optional: physical box length override (default: read from plot file)
        Real L_override = -1.0;
        pp.query("L", L_override);

        // Create the output directory on rank 0
        if (ParallelDescriptor::IOProcessor())
        {
            // Portable mkdir -p equivalent
            std::string path;
            for (char c : output_dir + "/")
            {
                path += c;
                if (c == '/')
                {
                    struct stat st;
                    if (stat(path.c_str(), &st) != 0)
                        mkdir(path.c_str(), 0755);
                }
            }
        }
        ParallelDescriptor::Barrier();

        // ---- Process each plot file ----
        for (const auto& pf_name : plotfiles)
        {
            Print() << "\n=== Processing: " << pf_name << " ===\n";

            PlotFileData pf(pf_name);

            // --- Grid size ---
            Box domain = pf.probDomain(0);
            int Nx = domain.length(0);
            int Ny = domain.length(1);
            int Nz = domain.length(2);
            if (Nx != Ny || Nx != Nz)
                Abort("Domain must be cubic. Got "
                    + std::to_string(Nx) + "x"
                    + std::to_string(Ny) + "x"
                    + std::to_string(Nz) + " in " + pf_name);
            const int N = Nx;

            // --- Physical box length ---
            Real L;
            if (L_override > 0.0)
            {
                L = L_override;
            }
            else
            {
                auto psize = pf.probSize();
                L = psize[0];
                if (std::abs(psize[1] - L) > 1.e-12 * L ||
                    std::abs(psize[2] - L) > 1.e-12 * L)
                    Abort("Domain must be cubic. probSize differs per axis.");
            }

            Print() << "  N = " << N << ",  L = " << L
                    << ",  time = " << pf.time() << "\n";

            // --- Find the component ---
            auto const& var_names = pf.varNames();
            int comp_idx = -1;
            for (int c = 0; c < static_cast<int>(var_names.size()); ++c)
                if (var_names[c] == comp_name) { comp_idx = c; break; }

            if (comp_idx < 0)
            {
                Print() << "  Available components:";
                for (const auto& n : var_names) Print() << "  " << n;
                Print() << "\n";
                Abort("Component '" + comp_name + "' not found in " + pf_name);
            }
            Print() << "  Component '" << comp_name
                    << "' found at index " << comp_idx << "\n";

            // --- Read the real-space R field ---
            MultiFab mf_R = pf.get(0, comp_name);

            // --- Forward FFT ---
            // Set up the real-space domain box (cell-centred, starts at 0)
            IntVect lo(0, 0, 0), hi(N-1, N-1, N-1);
            Box x_domain(lo, hi);

            FFT::R2C<Real, FFT::Direction::forward> r2c(x_domain);
            auto [cba, cdm] = r2c.getSpectralDataLayout();

            cMultiFab R_k(cba, cdm, 1, 0);
            R_k.setVal(GpuComplex<Real>{0.0, 0.0});

            r2c.forward(mf_R, R_k);

            // --- Normalise (matches RandomField::extract) ---
            // In extract(), after forward FFT:
            //   scalars_k.mult(1./norm/pow(N, 3.))
            // where norm = (sqrt(2*pi)/L)^3.
            // Because R_x stored in the plotfile equals backward_FFT(R_k)*norm
            // (see RandomField::derive), recovering R_k from R_x requires
            // dividing the raw FFT output by norm*N^3.
            const Real norm    = std::pow(std::sqrt(2.0 * M_PI) / L, 3.0);
            const Real inv_fac = 1.0 / (norm * std::pow(Real(N), 3.0));
            R_k.mult(inv_fac, 0, 1);

            // --- Compute and write the power spectrum ---
            // Build output filename from the plotfile basename
            std::string basename = pf_name;
            while (!basename.empty() && basename.back() == '/')
                basename.pop_back();
            auto slash = basename.rfind('/');
            if (slash != std::string::npos) basename = basename.substr(slash + 1);

            std::string out_path = output_dir + "/spectrum-" + basename + ".dat";

            compute_power_spectrum(R_k, N, L, out_path);
        }

        Print() << "\nDone.\n";
    }
    Finalize();
    return 0;
}
