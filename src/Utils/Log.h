#pragma once

#define LOG(x) \
    std::cout << x << std::endl;

#define LOGH1(x) LOG("\n===============================================================================\n" \
                     << "                           " << x)

#define LOGH2(x) LOG("\n" \
                     << "=> " << x)

#define LOGH3(x) LOG("- " << x)
