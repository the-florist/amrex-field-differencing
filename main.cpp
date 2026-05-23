/* PowerSpectrumExtractor
 * Reads one or more AMReX plot files, extracts a named real-space component
 * (default "R"), forward-FFTs it, applies the same physical normalisation used
 * in RandomField::extract(), and bins the power spectrum exactly as
 * RandomField::print_power_spectrum() does.
 *
 * Inputs (via ParmParse / AMReX inputs file or command-line key=value pairs):
 *
 *  --- Single-field mode (spectrum of one field) ---
 *   plotfiles    = plt00000 plt00100 ...  (explicit list of plot-file directories)
 *   plotfile_dir = /path/to/run/          (scan directory for all plot files)
 *
 *  --- Difference mode (spectrum of field1 - field2) ---
 *   plotfile_dir   = /path/to/run1/       (first source; or use plotfiles=...)
 *   plotfile_dir_2 = /path/to/run2/       (second source; or use plotfiles_2=...)
 *   Alternatively supply explicit lists:
 *   plotfiles_2  = plt00000 plt00100 ...
 *
 *  --- Shared options ---
 *   output_dir  = spectra/                (directory to write spectrum .dat files)
 *   component   = R                       (component name; default "R")
 *   L           = 1.0                     (physical box length; default: read from file)
 *
 * plotfiles and plotfile_dir may both be specified; results are merged and sorted.
 * The same applies to plotfiles_2 / plotfile_dir_2.
 * In difference mode both lists must contain the same number of entries;
 * they are paired in sorted order.
 *
 * Output (single mode):  <output_dir>/spectrum-<basename>.dat
 * Output (diff mode):    <output_dir>/spectrum-diff-<basename1>-vs-<basename2>.dat
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
        amrex::Error("compute_power_spectrum: Isotropic k axis check failed.");

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

        amrex::ParallelFor(bx, [=, &ps_map, &kcount]
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
            amrex::Abort("compute_power_spectrum: cannot open " + output_path);

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
// Helper: scan a directory for AMReX plot-file sub-directories.
// A directory is treated as a plot file iff it contains a "Header" file.
// The returned list is sorted alphabetically (= chronological for plt00000…).
// ---------------------------------------------------------------------------
std::vector<std::string> scan_plotfile_dir(const std::string& dir)
{
    namespace fs = std::filesystem;
    if (!fs::is_directory(dir))
        amrex::Abort("Not a directory: " + dir);

    std::vector<std::string> found;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (entry.is_directory() && fs::exists(entry.path() / "Header"))
            found.push_back(entry.path().string());
    }
    std::sort(found.begin(), found.end());
    return found;
}

// ---------------------------------------------------------------------------
// Helper: extract the final path component (basename) from a path string,
// stripping any trailing slashes first.
// ---------------------------------------------------------------------------
std::string path_basename(std::string p)
{
    while (!p.empty() && p.back() == '/') p.pop_back();
    auto slash = p.rfind('/');
    return (slash != std::string::npos) ? p.substr(slash + 1) : p;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // ---- Parse inputs ----
        ParmParse pp;

        // ---- Build the first (and possibly only) list of plot files ----
        Vector<std::string> plotfiles;

        int n_pf = pp.countval("plotfiles");
        if (n_pf > 0)
        {
            plotfiles.resize(n_pf);
            pp.getarr("plotfiles", plotfiles, 0, n_pf);
        }

        std::string plotfile_dir;
        if (pp.query("plotfile_dir", plotfile_dir))
        {
            auto found = scan_plotfile_dir(plotfile_dir);
            for (const auto& p : found) plotfiles.push_back(p);
        }

        if (plotfiles.empty())
            amrex::Abort("No plot files found. Specify plotfiles=... or plotfile_dir=...");

        // ---- Build the optional second list (activates difference mode) ----
        Vector<std::string> plotfiles_2;

        int n_pf2 = pp.countval("plotfiles_2");
        if (n_pf2 > 0)
        {
            plotfiles_2.resize(n_pf2);
            pp.getarr("plotfiles_2", plotfiles_2, 0, n_pf2);
        }

        std::string plotfile_dir_2;
        if (pp.query("plotfile_dir_2", plotfile_dir_2))
        {
            auto found = scan_plotfile_dir(plotfile_dir_2);
            for (const auto& p : found) plotfiles_2.push_back(p);
        }

        const bool diff_mode = !plotfiles_2.empty();

        if (diff_mode && plotfiles_2.size() != plotfiles.size())
            amrex::Abort("Difference mode: plotfile lists have different lengths ("
                + std::to_string(plotfiles.size()) + " vs "
                + std::to_string(plotfiles_2.size()) + "). "
                "They must be paired one-to-one.");

        if (diff_mode)
            Print() << "Running in DIFFERENCE mode ("
                    << plotfiles.size() << " pairs).\n";
        else
            Print() << "Running in SINGLE mode ("
                    << plotfiles.size() << " plot files).\n";

        // ---- Shared options ----
        std::string output_dir;
        pp.get("output_dir", output_dir);
        while (output_dir.size() > 1 && output_dir.back() == '/') output_dir.pop_back();

        std::string comp_name = "R";
        pp.query("component", comp_name);

        Real L_override = -1.0;
        pp.query("L", L_override);

        // Create the output directory on rank 0
        if (ParallelDescriptor::IOProcessor())
        {
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

        // ---- Helper: read grid size and box length from a PlotFileData ----
        // Returns N (cubic side length) and L (physical box length).
        auto read_grid = [&](const PlotFileData& pf, const std::string& name)
            -> std::pair<int, Real>
        {
            Box domain = pf.probDomain(0);
            int Nx = domain.length(0), Ny = domain.length(1), Nz = domain.length(2);
            if (Nx != Ny || Nx != Nz)
                amrex::Abort("Domain must be cubic in " + name + ": got "
                    + std::to_string(Nx) + "x" + std::to_string(Ny)
                    + "x" + std::to_string(Nz));
            Real L;
            if (L_override > 0.0)
            {
                L = L_override;
            }
            else
            {
                auto ps = pf.probSize();
                L = ps[0];
                if (std::abs(ps[1]-L) > 1.e-12*L || std::abs(ps[2]-L) > 1.e-12*L)
                    amrex::Abort("Domain must be cubic (probSize differs per axis) in " + name);
            }
            return {Nx, L};
        };

        // ---- Helper: find a named component or abort with a list of available names ----
        auto find_component = [&](const PlotFileData& pf, const std::string& pf_name) -> int
        {
            auto const& vn = pf.varNames();
            for (int c = 0; c < static_cast<int>(vn.size()); ++c)
                if (vn[c] == comp_name) return c;
            Print() << "  Available components in " << pf_name << ":";
            for (const auto& n : vn) Print() << "  " << n;
            Print() << "\n";
            amrex::Abort("Component '" + comp_name + "' not found in " + pf_name);
            return -1; // unreachable
        };

        // ---- Process each pair (or single file) ----
        for (int idx = 0; idx < static_cast<int>(plotfiles.size()); ++idx)
        {
            const std::string& pf_name  = plotfiles[idx];
            Print() << "\n=== Processing: " << pf_name;
            if (diff_mode) Print() << "  minus  " << plotfiles_2[idx];
            Print() << " ===\n";

            // --- Open first plot file ---
            PlotFileData pf1(pf_name);
            auto [N, L] = read_grid(pf1, pf_name);
            find_component(pf1, pf_name); // validates; actual read uses name directly

            Print() << "  N = " << N << ",  L = " << L
                    << ",  time = " << pf1.time() << "\n";

            // --- Read the component from plot file 1 ---
            MultiFab mf_field = pf1.get(0, comp_name);

            // --- Difference mode: subtract component from plot file 2 ---
            if (diff_mode)
            {
                PlotFileData pf2(plotfiles_2[idx]);
                auto [N2, L2] = read_grid(pf2, plotfiles_2[idx]);
                find_component(pf2, plotfiles_2[idx]);

                if (N2 != N)
                    amrex::Abort("Grid size mismatch between pair "
                        + std::to_string(idx) + ": N=" + std::to_string(N)
                        + " vs N=" + std::to_string(N2));
                if (std::abs(L2 - L) > 1.e-10 * L)
                    amrex::Abort("Box size mismatch between pair "
                        + std::to_string(idx) + ": L=" + std::to_string(L)
                        + " vs L=" + std::to_string(L2));

                Print() << "  time2 = " << pf2.time() << "\n";

                // Read the component from plotfile 2.
                // Remap it onto mf_field's box decomposition (handles different
                // MPI layouts between the two runs) then subtract in-place.
                MultiFab mf2 = pf2.get(0, comp_name);
                MultiFab mf2_r(mf_field.boxArray(), mf_field.DistributionMap(), 1, 0);
                mf2_r.ParallelCopy(mf2, 0, 0, 1);
                MultiFab::Subtract(mf_field, mf2_r, 0, 0, 1, 0);

                Print() << "  Difference field computed.\n";
            }

            // --- Forward FFT of the (possibly differenced) field ---
            IntVect lo(0, 0, 0), hi(N-1, N-1, N-1);
            Box x_domain(lo, hi);

            FFT::R2C<Real, FFT::Direction::forward> r2c(x_domain);
            auto [cba, cdm] = r2c.getSpectralDataLayout();

            cMultiFab field_k(cba, cdm, 1, 0);
            field_k.setVal(GpuComplex<Real>{0.0, 0.0});

            r2c.forward(mf_field, field_k);

            // --- Normalise (matches RandomField::extract) ---
            const Real norm    = std::pow(std::sqrt(2.0 * M_PI) / L, 3.0);
            const Real inv_fac = 1.0 / (norm * std::pow(Real(N), 3.0));
            field_k.mult(inv_fac, 0, 1);

            // --- Build output filename ---
            std::string basename1 = path_basename(pf_name);
            std::string out_path;
            if (diff_mode)
            {
                std::string basename2 = path_basename(plotfiles_2[idx]);
                out_path = output_dir + "/spectrum-diff-"
                         + basename1 + "-vs-" + basename2 + ".dat";
            }
            else
            {
                out_path = output_dir + "/spectrum-" + basename1 + ".dat";
            }

            // --- Compute and write the power spectrum ---
            compute_power_spectrum(field_k, N, L, out_path);
        }

        Print() << "\nDone.\n";
    }
    amrex::Finalize();
    return 0;
}
