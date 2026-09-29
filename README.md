# ClearWater6.1

[Live demo](https://samg-coder.github.io/ClearWater6.1/) ? [Build and deployment](https://github.com/SamG-Coder/ClearWater6.1/actions/workflows/pages.yml)

![Shallow water with FFT drag interaction](docs/preview.png)

An original CUDA WebShader water study with FFT waves, clear shallow-water refraction, sunlight caustics, a procedural seabed, and a GPU-resident fly camera. The water code was written for this project; it does not use Clearwater or other demo source. Only the CUDA WebShader compiler and runtime are vendored from [CUDA WebShader](https://github.com/SamG-Coder/cuda-webshader) under its MIT license.

## Run

```powershell
npm install
npm run build
npm start
```

Open http://127.0.0.1:5191 in a browser with WebGPU enabled. `npm run build` compiles every kernel ahead of time, so the browser loads generated artifacts rather than recompiling CUDA source. `dist/` is a standalone static site, including the CUDA source and runtime.

## Explore

Hold the left mouse button and drag across the water to apply force. Right-drag to look, or click **Fly camera** to capture the mouse. **WASD** flies in the viewing direction; **Q/E** lowers/raises the camera; **Shift** boosts speed. Scroll changes speed. **Esc** releases the mouse. Arrow keys look, **H** hides the panel, and **Reset** restores the shallow-water view. The camera stays above the surface; this is an above-water water study.

Depth, wave energy, wind, exposure, and render resolution are adjustable. **Shallows** and **Open water** set depth, wave energy, and camera presets. The caustic and normal views expose the rendering components.

## CUDA implementation

All simulation, camera integration, ray generation, lighting, seabed materials, optics, tone mapping, and pixel packing live in `src/water.cu`. JavaScript handles input events, UI, resource creation, dispatches, and copying the GPU output to the canvas. There are no downloaded textures, Three.js, WebGL, or handwritten WGSL shaders in the water implementation.

- Two deterministic 128 × 128 complex spectral cascades span 6 m and 96 m. A wind-aligned Phillips-style spectrum evolves with finite-depth gravity-wave dispersion, `omega² = g k tanh(k depth)`.
- Hermitian spectra and a separable radix-2 inverse FFT produce real heights. Resolved finite differences produce normals. The deliberately unnormalized inverse FFT uses source amplitudes calibrated in world units.
- The renderer intersects the height field and refracts view rays using water's refractive index. Rays intersect a gently undulating procedural seabed with sand ripples and scattered stones.
- Fresnel reflection, per-channel Beer-Lambert absorption, water scattering, and sun highlights shade the surface.
- A 256 × 256 sunlight map follows refracted rays to the mean-depth plane and estimates the photon mapping Jacobian. Area compression produces caustic focusing, with slightly different RGB indices of refraction. The map follows the short-wave cascade; the long cascade contributes surface shape and normals. This keeps the caustic tile periodic and bounded in cost.
- A moving left-button drag is projected onto the water on the GPU. A Gaussian pressure path injects vertical velocity into a third complex spectral field, evolved with an analytic damped gravity-wave oscillator before the same inverse FFT. The wake changes surface height, normals, reflection and view-ray refraction, and persists after release. Stationary clicks and hover inject no force. The caustic photon map uses the background short-wave cascade; it does not include the interaction cascade. Interaction waves repeat every 24 m.
- Camera state remains in a GPU buffer. CPU inputs supply motion and look axes. The live frame loop never downloads camera, simulation, or pixel buffers.

This is a height-field approximation. It has no overturning breakers, underwater camera, shoreline wetting, or volumetric multiple scattering. Forward photon transport accumulates overlapping refracted rays; the finite light-map resolution bounds the smallest visible caustic features. Spectral fields repeat at their patch lengths, and the caustic tile repeats every 6 m.

## Validation

Start the local server, then run `npm test`. Run `npm run test:force` to exercise actual held mouse drags, hover and stationary-click negative controls, persistent wakes, and right-drag camera controls. Browser checks cover a direct complex Fourier-series oracle on three known modes, Hermitian residual, finite heights and lighting, a flat-water caustics negative control, changing image output over time, GPU camera movement, zero readbacks during live rendering, and full-compute GPU timestamp measurements at balanced resolution and 1080p. Screenshots and numerical results are written to `captures/`.

`waterLab` exposes `inspect()`, `fftTest()`, `flatCausticsTest()`, `seek(seconds)`, `benchmark(samples)`, `pause()`, and `resume()` for explicit diagnostics. Readbacks occur only when these inspection functions are requested. GPU timings include camera integration, spectrum evolution, FFT, normals, caustics, and rendering; they exclude the final texture copy and browser display. They are not an end-to-end FPS guarantee.

The browser executes CUDA-subset source compiled to WGSL on WebGPU. This project has not been validated as a native CUDA executable.

## GitHub Pages

Pushes to `main` run JavaScript syntax checks, compile all ten CUDA kernels into WebGPU artifacts, build the static site, and deploy `dist/` to GitHub Pages. Pull requests run the build without deploying. The workflow supports manual runs. Hosted Actions performs compilation and packaging; hardware WebGPU browser tests are run locally with `npm test` and `npm run test:force`.

## License

MIT. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
