#include "RadiositySolver.h"
#include "Scene.h"

int main() {
    auto lightmapSize = uvec2(128);
    auto scene = makeRef<Scene>();
    scene->faces = loadMesh("resources/cornell_box.obj");

    RadiositySolver solver(lightmapSize);
    solver.initialize(scene);
    solver.solve(0.1f);

    solver.lightmap().save("lightmap.hdr");

    return EXIT_SUCCESS;
}
