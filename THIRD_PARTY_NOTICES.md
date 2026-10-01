# Third-party software

`vendor/webcuda/compiler/` and `vendor/webcuda/runtime/` contain CUDA WebShader code from [SamG-Coder/cuda-webshader](https://github.com/SamG-Coder/cuda-webshader), licensed under MIT. The license and copyright notice are preserved in `vendor/WEBCUDA-LICENSE`.

The spherical weather implementation in `src/water.cu` adapts wind-energy memory, world-space rain planes, and rain-impact shading from [SamG-Coder/clearwater](https://github.com/SamG-Coder/clearwater), `src/clearwater.cu` at `daff9a9853c8c18179492fb95f0601e4d8c287d3`, as requested by the project owner. Its MIT copyright and license are preserved in `vendor/CLEARWATER-LICENSE`. The moving pressure systems, spherical weather sampling, rotation and globe integration were implemented for ClearWater6.1. The original planar storm, tornado solver, geometry and interface were not imported.

The interface optionally loads DM Sans and Manrope from Google Fonts. System sans-serif fallbacks are used when the font service is unavailable.

Playwright is a development dependency used for browser validation; it is not included in the deployed site.
