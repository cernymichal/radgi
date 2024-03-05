#include <argh.h>

#include "RadiositySolver.h"
#include "Scene.h"

auto constexpr USAGE =
    R"(Usage: radgi <input_file> [-o output_file] [-r resolution] [-t threshold] [-i iterations] [-e radius]

Options:
    <input_file>                    Input scene in .obj format

	-o, --output <output_file>      Output file for the lightmap (output.hdr by default)

	-r, --resolution <resolution>   Lightmap resolution (128 by default)

	-t, --threshold <threshold>     Residue threshold for terminating (0.25 by default)
                                    This is the default mode - progressive refinement.
                                    The scene is solved until the maximum patch residue is below the threshold.

	-i, --iterations <iterations>   Number of iterations through the whole scene
     			                    When set, -t is ignored and the scene is solved uniformly.

	-p, --padding <radius>          Extrapolate lightmap regions with the given radius in pixels (2 by default)

    -h, --help                      Print this help message


Example:
	radgi resources/cornell_box.obj -o lightmap.hdr -r 256 -t 0.3)";

int main(int argc, char* argv[]) {
    argh::parser cmdl(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

    if (!cmdl(1) || cmdl[{"-h", "--help"}]) {
        LOG(USAGE);
        return EXIT_FAILURE;
    }
    auto inputFile = cmdl[1];

    std::string outputFile;
    cmdl({"-o", "--output"}, "output.hdr") >> outputFile;

    uint32_t resolution;
    cmdl({"-r", "--resolution"}, 128) >> resolution;

    float threshold;
    cmdl({"-t", "--threshold"}, 0.25f) >> threshold;
    bool useProgressive = true;

    uint32_t iterations;
    if (cmdl({"-i", "--iterations"}) >> iterations)
        useProgressive = false;

    uint32_t paddingRadius;
    cmdl({"-p", "--padding"}, 2) >> paddingRadius;

    auto lightmapSize = uvec2(resolution);
    auto scene = makeRef<Scene>();
    scene->faces = loadMesh(inputFile);

    RadiositySolver solver(lightmapSize);
    solver.initialize(scene);

    LOG("Solving");

    auto start = std::chrono::high_resolution_clock::now();
    if (useProgressive)
        solver.solveProgressive(threshold);
    else
        solver.solveUniform(iterations);
    auto finish = std::chrono::high_resolution_clock::now();

    LOG(std::format("Solved in {:.2f}s", std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() / 1000.0));

    LOG("Adding padding to the lightmap");
    solver.addPadding(paddingRadius);

    solver.lightmap().save(outputFile);

    return EXIT_SUCCESS;
}
