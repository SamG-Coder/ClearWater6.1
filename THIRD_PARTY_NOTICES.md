# Third-party software

`vendor/webcuda/compiler/` and `vendor/webcuda/runtime/` contain CUDA WebShader code from [SamG-Coder/cuda-webshader](https://github.com/SamG-Coder/cuda-webshader), licensed under MIT. The license and copyright notice are preserved in `vendor/WEBCUDA-LICENSE`.

The water simulation and rendering in `src/water.cu` are original to this project.

The interface optionally loads DM Sans and Manrope from Google Fonts. System sans-serif fallbacks are used when the font service is unavailable.

Playwright is a development dependency used for browser validation; it is not included in the deployed site.
