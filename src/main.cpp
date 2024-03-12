#include <argh.h>

#include "RadiositySolver.h"
#include "Scene.h"

auto constexpr USAGE =
    R"(Usage: radgi <input_file> [-o output_file] [-r resolution] [-t threshold] [-i iterations] [-e radius]

Options:
    <input_file>                    Input scene in .obj format

	-o, --output <output_file>      Output file for the lightmap (output.exr by default)

	-r, --resolution <resolution>   Lightmap resolution (128 by default)

	-t, --threshold <threshold>     Residue threshold for terminating (0.1 by default)
                                    This is the default CPU mode - progressive shooting.
                                    The scene is solved until the maximum patch residue is below the threshold.

	-i, --iterations <iterations>   Number of gathering iterations through the whole scene
     			                    When set, -t is ignored and the scene is solved by gathering instead of shooting.

	-d, --dilation <radius>         Extrapolate lightmap islands with the given radius in pixels (2 by default)

    -g, --gpu                       Use GPU acceleration (not implemented yet) (4 iterations by default)
                                    Only the gathering mode is supported on the GPU. -t is ignored.

    -h, --help                      Print this help message


Example:
	radgi resources/cornell_box/cornell_box.obj -o lightmap.exr -r 256 -t 0.01)";

int main(int argc, char* argv[]) {
    argh::parser cmdl(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

    if (!cmdl(1) || cmdl[{"-h", "--help"}]) {
        LOG(USAGE);
        return EXIT_FAILURE;
    }
    auto inputFile = cmdl[1];

    std::string outputFile;
    cmdl({"-o", "--output"}, "output.exr") >> outputFile;

    uint32_t resolution;
    cmdl({"-r", "--resolution"}, 128) >> resolution;

    float threshold;
    cmdl({"-t", "--threshold"}, 0.1f) >> threshold;
    bool useShooting = true;

    bool useGPU = cmdl[{"-g", "--gpu"}];

    uint32_t iterations = 4;
    if (cmdl({"-i", "--iterations"}) >> iterations || useGPU)
        useShooting = false;

    uint32_t dilationRadius;
    cmdl({"-d", "--dilation"}, 2) >> dilationRadius;

    auto lightmapSize = uvec2(resolution);
    auto scene = makeRef<Scene>();
    scene->faces = loadMesh(inputFile);

    RadiositySolver solver(lightmapSize);
    solver.initialize(scene);

// #define OUTPUT_PATCH_GEOMETRY
#ifdef OUTPUT_PATCH_GEOMETRY
    saveMesh("patches.obj", solver.createPatchGeometry());
#endif

    LOG("Solving");

    std::chrono::steady_clock::time_point start, finish;

    if (useGPU) {
        LOG("Using GPU mode");

        start = std::chrono::high_resolution_clock::now(),
        finish = std::chrono::high_resolution_clock::now();
    }
    else {
        LOG("Using CPU mode");

        start = std::chrono::high_resolution_clock::now();
        if (useShooting)
            solver.solveShooting(threshold);
        else
            solver.solveGathering(iterations);
        finish = std::chrono::high_resolution_clock::now();
    }

    LOG(std::format("Solved in {:.2f}s", std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() / 1000.0));

    LOG("Dilating the lightmap");
    solver.dilateLightmap(dilationRadius);

    solver.lightmap().save(outputFile, true);

    return EXIT_SUCCESS;
}
