#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gsd.h"

int main(int argc, char** argv) // NOLINT
    {
    const size_t n_keys = 40000;
    const size_t max_frames = 100;
    std::vector<char> data;

    const size_t key_size = static_cast<const size_t>(1024) * static_cast<const size_t>(1024);

    std::vector<std::string> names;
    for (size_t i = 0; i < n_keys; i++)
        {
        std::ostringstream s;
        s << "quantity/" << i;
        names.push_back(s.str());
        }
        
    auto starttime = std::chrono::high_resolution_clock::now();

    gsd_handle handle;
    gsd_open(&handle, "test.gsd", GSD_OPEN_READONLY);
    size_t const n_frames = gsd_get_nframes(&handle);
    size_t n_read = n_frames;
    if (n_read > max_frames)
        {
        n_read = max_frames;
        }

    std::cout << "Reading test.gsd with: " << n_keys << " keys and " << n_frames << " frames."
              << '\n';

    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t frame = 0; frame < n_read; frame++)
        {
        for (auto const& name : names)
            {
            const gsd_index_entry* e;
            e = gsd_find_chunk(&handle, frame, name.c_str());
            if (data.empty())
                {
                data.resize(e->N * e->M * gsd_sizeof_type((gsd_type)e->type));
                }
            gsd_read_chunk(&handle, data.data(), e);
            }
        }

    auto t2 = std::chrono::high_resolution_clock::now();

    gsd_close(&handle);

    auto endtime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> const time_span
        = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    double const time_per_key = time_span.count() / double(n_keys) / double(n_read);

    std::chrono::duration<double> abs_time_span
        = std::chrono::duration_cast<std::chrono::duration<double>>(endtime - starttime);
    double abs_time = time_span.count();

    const double total_mb
        = double(n_keys * n_read * key_size * 8 + static_cast<const size_t>(32) * static_cast<const size_t>(2))
          / 1000000;

    const double us = 1e-6;
    std::cout << "Sequential read time: " << time_per_key / us << " microseconds/key." << std::endl;

    std::cout << "Total time required: " << abs_time << " seconds for " << total_mb << " MB or " << total_mb/1000 << " GB" << std::endl; 
    std::cout << "!!!!!!!!!!!! Finished Reading to File !!!!!!!!!!!!" << std::endl;

    }
