# ClearWater6.1

[Live demo](https://samg-coder.github.io/ClearWater6.1/) ? [Build and deployment](https://github.com/SamG-Coder/ClearWater6.1/actions/workflows/pages.yml)

![Shallow water with FFT drag interaction](docs/preview.png)

A CUDA WebShader ocean at Earth scale, with FFT waves, clear shallow-water refraction, sunlight caustics, a procedural seabed, global weather, and a GPU-resident fly camera. The owner explicitly requested adapting weather techniques from `D:\ClearWater`; wind-energy memory and rain rendering now use those ideas. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact source and license. The compiler and runtime come from [CUDA WebShader](https://github.com/SamG-Coder/cuda-webshader).

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
- A 512 × 512 PC sunlight map (256 × 256 in Mobile and Performance) follows refracted rays to the mean-depth plane and estimates the photon mapping Jacobian. Area compression produces caustic focusing, with slightly different RGB indices of refraction. The map follows the short-wave cascade; the long cascade contributes surface shape and normals. This keeps the caustic tile periodic and bounded in cost.
- A moving left-button drag is projected onto the water on the GPU. A Gaussian pressure path injects vertical velocity into a third complex spectral field, evolved with an analytic damped gravity-wave oscillator before the same inverse FFT. The wake changes surface height, normals, reflection and view-ray refraction, and persists after release. Stationary clicks and hover inject no force. The caustic photon map uses the background short-wave cascade; it does not include the interaction cascade. Interaction waves repeat every 24 m.
- Camera state remains in a GPU buffer. CPU inputs supply motion and look axes. The live frame loop never downloads camera, simulation, or pixel buffers.

This is a height-field approximation. It has no overturning breakers, underwater camera, shoreline wetting, or volumetric multiple scattering. Forward photon transport accumulates overlapping refracted rays; the finite light-map resolution bounds the smallest visible caustic features. Spectral fields repeat at their patch lengths, and the caustic tile repeats every 6 m.

## Validation

Start the local server, then run `npm test`. Run `npm run test:force` to exercise actual held mouse drags, hover and stationary-click negative controls, persistent wakes, and right-drag camera controls. Browser checks cover a direct complex Fourier-series oracle on four known modes, Hermitian residual, finite heights and lighting, a flat-water caustics negative control, changing image output over time, GPU camera movement, GPU-resident simulation with small asynchronous UI telemetry, and full-compute GPU timestamp measurements at balanced resolution and 1080p. Screenshots and numerical results are written to `captures/`.

`waterLab` exposes `inspect()`, `fftTest()`, `flatCausticsTest()`, `seek(seconds)`, `benchmark(samples)`, `pause()`, and `resume()` for explicit diagnostics. Simulation fields stay GPU-resident; a 160-byte asynchronous camera/weather telemetry read every 15 frames updates the UI. Detailed field readbacks occur only for diagnostics. GPU timings include camera integration, spectrum evolution, FFT, normals, caustics, and rendering; they exclude the final texture copy and browser display. They are not an end-to-end FPS guarantee.

The browser executes CUDA-subset source compiled to WGSL on WebGPU. This project has not been validated as a native CUDA executable.

## GitHub Pages

Pushes to `main` run JavaScript syntax checks, compile all CUDA kernels into WebGPU artifacts, build the static site, and deploy `dist/` to GitHub Pages. Pull requests run the build without deploying. The workflow supports manual runs. Hosted Actions performs compilation and packaging; hardware WebGPU browser tests are run locally with `npm test` and `npm run test:force`.

## License

MIT. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Mobile

Touch devices automatically select **Mobile / Auto** and start with the settings panel closed. The profile targets 30 rendered frames per second and limits the longest render edge to 960 pixels. Sustained slow frame completion reduces resolution automatically, down to half scale; recovered performance gradually restores it. High phone pixel ratios cannot multiply the render workload without a limit. Simulation and camera/lighting/render math remain in CUDA.

The mobile sunlight pass uses 256 × 256 rays and a single water index of refraction rather than the PC quality 1024 × 1024 RGB ray set (512 × 512 in Performance). It keeps focused caustics while reducing photon splat work by a factor of 48 compared with PC quality. The 256 × 256 light map, all three 128 × 128 FFT fields, force propagation, refraction, and seabed remain enabled. Random wind spectra are generated once and rebuilt only when wind changes. Kernel artifacts download concurrently. Rendering pauses while the page is hidden, and touch cancellation clears movement inputs.

Screen footprint controls the fine sand grain, gravel visibility, and sunlight highlight width, suppressing subpixel shimmer while preserving nearby detail. Short-wave displacement and normals fade together in the distance to stabilize the horizon. Distant gravel searches are skipped once their detail cannot be resolved.

Phones start in **Mode: Look**: drag anywhere on the water to look around. The left **MOVE** joystick controls forward/backward and sideways flight and works while another finger drags to look. It has analog speed, a center dead zone, and automatic centering on release/cancellation. The − / + buttons lower or raise the camera. Switch to **Mode: Water** to apply force by dragging. Open **Settings** for water, wind, exposure and resolution controls. Desktop mouse and keyboard controls are also available.

Run `npm run test:mobile` with the local server running. The test uses browser touch events in Pixel 7 and iPhone 13 viewport emulation and checks force injection, look/flight controls, touch cancellation, orientation resizing, spectrum caching, and illumination normalization. The iPhone layout supports screen safe areas in portrait and landscape. This does not emulate a phone GPU, thermal throttling, or battery use. Physical Android/iPhone performance and Safari rendering remain unverified. A WebGPU-capable browser/device is required; this project has no WebGL fallback. Safari 26 introduced WebGPU on iPhone; older Safari versions cannot run this renderer. Use the HTTPS Pages link on a phone.

## Optimization without reducing quality

The 128² FFTs now use two shared-memory axis passes rather than fourteen global-memory stages. Cached twiddle factors preserve the same spectrum. Camera basis vectors are calculated once per frame, untouched force coefficients skip oscillator math, and a conservative gravel bound skips stones whose original coverage is exactly zero. Fixed lighting powers use float multiplication chains instead of the compiler runtime's software double-precision integer-power path. Resolution, FFT dimensions, sunlight ray count, and filtered caustics are unchanged.

Run `npm run test:optimization` with the server running. It compares mobile frames against this project's own published commit `6860d41`, checks independent Fourier oracles for both FFT implementations, benchmarks the same resolution, and checks that slow displayed FPS is not clamped to ten. The mobile comparison now requires every pixel channel to match exactly. Desktop regression comparisons select Performance in both versions so that intentional PC optics upgrades do not hide regressions in the retained lightweight path. Desktop timings do not predict Samsung A34 FPS.

Depth-dependent dispersion and force envelopes are now cached and refreshed only when depth changes. Height-only intersection queries avoid evaluating unused slopes; underwater pixels skip unused direct-sky shading. Positive lighting powers use native float log2/exp2 math, so the render shader contains no software double-precision helpers. Mobile photons occupy a tightly packed single-channel working region, and the tent filter reads a shared 10×10 tile per 8×8 workgroup (100 global reads instead of 576). Compute and canvas transfer share one command submission. These changes retain the same resolution, FFT dimensions, ray count, filter weights, and shading parameters.

To choose an older baseline, set `WATER_REFERENCE_COMMIT` before running `npm run test:optimization`. The script also checks high-wind shallow water and low-wind deep water. Reference sources come only from this repository's own Git history.
The frequency cache uses packed float arrays (320 KiB), so background frequency reads are contiguous rather than padded float4 records. The force field is exactly zero before the first water drag; its FFT and oscillator dispatches are deferred until that drag. All three full-size fields remain allocated, and after activation the force field continues evolving every frame, including after release. No decay threshold or reduction in wave detail is used.

The latest math reuse pass shares the surface/view dot product between Fresnel, reflection and refraction, the distance fade between normals and caustics, the pixel footprint between seabed filtering and highlights, and the vertical refraction denominator across bed intersections. Gravel outlines reuse one angular calculation. The existing GPU camera kernel computes depth-only optical attenuation once per frame in spare basis components, without another allocation or dispatch. Against `dbfa572`, six reference scenes were byte-identical. At 448 × 912 on desktop NVIDIA/Edge, 80 samples measured median GPU compute time of 0.060448 ms before and 0.059648 ms after (about 1.3% lower); this modest result is not a physical-phone benchmark.

The next bandwidth pass skips all five bilinear pressure-field queries per water pixel until the first force gesture, then immediately uses the full field and continues its waves after release. Surface intersection and normal arithmetic are unchanged. Rendering uses 32 × 2 groups instead of 8 × 8, keeping 64 lanes while improving horizontal pixel access. Six reference frames against `630da51` remained byte-identical; a sky drag additionally checks that activating a zero pressure field leaves pixels unchanged. At 448 × 912, desktop NVIDIA/Edge median compute time measured 0.059552 ms before and 0.052096 ms after (12.5% lower). Two alternating 400-sample experiments measured the strip layout at 0.051968–0.052064 ms versus 0.053376 ms for square groups with the same skipped-pressure shader. These are desktop measurements; the skipped-field benefit applies before the first force gesture, while strip layout applies throughout. No resolution, ray count, material, or simulation fidelity was reduced.

Mobile lighting now stores one float per resolved texel and allocates only the single-channel photon working region. Together these buffers use 524,304 bytes (512 KiB plus a 16-byte inactive RGB binding), down from 2 MiB. The renderer interpolates monochrome light once and expands the result to RGB; desktop still uses full RGB dispersion. Buffer bindings are replaced when switching profiles, and flat-water diagnostics use isolated photon buffers. Random-cell normalization uses exact binary scaling by 2^-24; supported photon normalizations use native division by powers of two. Sun-disk powers are skipped below a .99 direction dot product, where their value is already below the float range. Seven scenes, including a camera aimed at the sun, matched the preceding release byte for byte. The mobile viewport benchmark measured 0.052256 ms before and 0.048512 ms after (7.2% lower median GPU compute time on desktop NVIDIA/Edge). A shared photon-sampling tile was tested and discarded because repeated runs were slightly slower. Set `WATER_DESKTOP=1` before running `npm run test:optimization` to also check the full RGB profile; actual phone performance remains unmeasured locally.

The same seven scenes were also byte-identical in the full RGB desktop profile at 1152 × 720. Its median compute time measured 0.125120 ms before and 0.118176 ms after. Mobile tests additionally verify the 524,304-byte lighting allocation, both profile-switch directions, flat-water energy conservation, joystick/screen-look input, cancellation, and persistent pressure waves.

PC quality now uses four fixed spatial samples per output pixel in Balanced, 1080p, 1440p, and 4K modes. Each sample evaluates the water/refraction/material shading with the correct half-pixel footprint, and their linear radiance is averaged before tone mapping. This improves fine-detail sampling and edge antialiasing without temporal blending. Performance and Mobile modes retain one center sample. Four-sample rendering uses the existing render dispatch and output buffer; FFT and lighting passes still run once per frame. Run `npm run test:pc` for before/after screenshots, sampling-mode checks, unchanged simulation/light checks, and GPU timings. Its desktop NVIDIA/Edge run at 1152 × 720 measured approximately 0.12 ms for one sample and 0.32 ms for four; compute timings exclude canvas transfer/display and do not guarantee every PC reaches 60 FPS. Zero-result highlight/gravel guards were tested but omitted because they gave no reliable speedup. Mobile output remains byte-identical in the seven reference scenes. `waterLab.setPixelSamples(0|1|4)` is an explicit diagnostic override; 0 restores automatic selection and Mobile always remains single-sample.

The PC multisample path has a separate `render_pc` entry point. Mobile and Performance dispatch the one-sample `render` shader, which excludes the PC sampling loop. Both reuse the same CUDA shading and pixel-packing functions, and each frame still has one render dispatch and one command submission.

PC quality verification also sets the actual GPU framebuffer to 2560 × 1440 and 3840 × 2160, confirms those dimensions and four spatial samples, and records 120 GPU timing samples plus 40 live frame-completion samples at each resolution. The completion measurement includes JavaScript encoding and the canvas copy through GPU completion, but excludes physical display latency. High-resolution screenshots are saved in `captures/`.

At actual 2560 × 1440 with four samples, 120 samples measured 1.148448 ms median / 1.274784 ms p95 GPU compute. At 3840 × 2160, they measured 2.481984 ms median / 2.789152 ms p95. Forty live frame-completion samples measured 3.3 / 4.0 ms at 1440p and 3.8 / 4.4 ms at 4K (median / p95), with approximately 60 Hz frame scheduling. These fixed-time shallow-water measurements used desktop NVIDIA Blackwell with Edge; they are not physical-phone results or universal PC FPS guarantees. Resolution choices cap render width at the selected value and the viewport physical width, so use fullscreen on a matching display to render at the full selected resolution.

## PC surface and sunlight quality

Balanced, 1080p, 1440p and 4K reconstruct the FFT heights with a periodic cubic B-spline. Height, normals and curvature remain continuous across cells. Two separable seven-tap coefficient passes approximately invert the spline smoothing operator through third order, avoiding the wave attenuation of an unprefiltered spline. The Fourier simulation itself remains unchanged. All view samples and sunlight rays share these coefficients. The two 768 KiB coefficient buffers exist only in PC quality profiles; Mobile and Performance keep their existing reconstruction and allocations.

PC quality uses a 512² RGB light map and 1024² photon paths per channel, four times the previous texel and photon counts. RGB refractions share one height/normal reconstruction per photon. A periodic, energy-preserving tent filter suppresses the regular splat lattice; shared-memory tiles limit global reads. PC light lookup now aligns with photon texel centers. The renderer uses six water intersection refinements and four seabed refinements, retaining four spatial samples per pixel. Mobile math, maps, ray counts and sampling remain unchanged.

The first interpolating cubic trial exposed straight bands in caustics because its curvature changed at cell boundaries. It was replaced with the continuous-curvature spline and the photon reconstruction filter. `npm run test:pc` captures a 900 × 700 close-up from the actual 4K render, and additionally benchmarks a downward-facing 4K scene with maximum wind/energy and active pressure waves. Its independent Fourier oracle checks reconstructed heights and analytic derivatives at 64 off-grid positions, including periodic boundaries. RMS height error measured 0.00274928 m for bilinear reconstruction versus 0.00001928 m for the spline; slope error measured 0.0611777 versus 0.0009808. These are results for the specified analytic test surface, not an accuracy bound for every spectrum.

On desktop NVIDIA Blackwell / Edge, 120-sample median / p95 GPU compute times were 2.826112 / 3.471296 ms at 2560 × 1440, and 5.894080 / 6.452256 ms at 3840 × 2160. Forty frame-completion samples measured median / p95 of 5.1 / 6.5 ms and 7.9 / 9.5 ms respectively. The 4K stress scene measured 8.459136 / 8.759552 ms GPU compute. GPU timing excludes canvas transfer/display; completion includes encoding and canvas copy through queue completion. Phone hardware and universal PC performance are not inferred from these results. Mobile reference scenes against `d1d6906` remain byte-identical.

## PC seabed relief

PC quality now intersects actual procedural sand ridges and small stones. Sand direction, spacing and amplitude vary smoothly in world space; analytic derivatives drive its underwater diffuse lighting. Two coarse bed refinements are followed by two relief refinements, so the geometry changes refraction and material placement. Stone placement is deterministic and sparse, with density evaluated at each object centre. Unequal convex faces and rounded joins form partly buried chips with tilted tops, low relief, mineral mottling and muted highlights. The projected contact shadows are an approximation using the refracted mean sun direction, not traced occlusion. All material and intersection code remains in CUDA; no textures or models are added.

The first domed gravel trial looked like raised beads and was rejected visually. The final stones use much lower relief (up to about 9 mm in the diagnostic patch), fewer objects and irregular faceted outlines. Candidate bounds reject irrelevant stones before density and shape calculations. Mobile and Performance retain the previous seabed code. `waterLab.bedTest()` compares analytic bed slopes with centred finite differences at 4096 points, covering sand and gravel. `npm run test:pc` saves a dedicated material close-up and measures actual frame completion in the heavy 4K pressure/wind scene as well as the standard resolutions.

The PC material shader is separate from the retained Mobile/Performance shading function. An explicit PC single-sample diagnostic entry shares the upgraded material code with PC multisampling, so sampling comparisons evaluate the same optics. No PC relief calculations enter the lightweight shader dependency graph.

The optimization verifier also checks that the lightweight render shader matches release `6860d41` after normalizing generated temporary names and whitespace. Local NVIDIA/Edge measurements for this material pass were approximately 4.1 ms at 1440p and 8.4 ms at 4K GPU compute. The heavy 4K test was around 12 ms GPU compute and 13-14 ms frame completion, so further PC quality increases should follow another optimization pass. These are local measurements, not a physical-phone or universal PC guarantee.


## FFT-driven shallow sand

PC quality profiles retain a small 128 by 128 sediment phase field on the GPU (256 KiB). A CUDA transport pass integrates the resolved long-wave FFT, depth-attenuated short waves and interactive pressure waves. Its response fades smoothly with depth and stops at 3 metres. Integration uses elapsed time, pauses with the simulation, and keeps its state between frames. Continuous-curvature sampling warps the sand ridges; the same deformation enters bed intersections and analytic lighting normals. Gravel remains anchored.

This is a visual sediment transport approximation, not a sediment mass conservation or full coastal morphology solver. Mobile and Performance retain their existing shader and do not allocate the sediment field unless a PC quality profile is selected. No extra FFT is required. The PC verifier checks actual FFT forcing against flat-water, deep-water and zero-timestep controls, then compares sand before and after live evolution with water frozen at the same simulation instant.


## Local interaction and regional wave variation

Touch/mouse pressure uses a single world-space 24 metre FFT domain. Its full-strength centre extends 7 metres in each direction and a quintic taper reaches zero at the 12 metre boundary, including the taper derivative in surface normals. Coordinates are relative to the interaction origin; neighboring periodic copies are never sampled. When a new brush position moves more than 6 metres from the domain centre, the solver recentres and replaces its previous wake. This keeps one pressure FFT and constant storage; it does not retain unlimited old wakes across the world.

Background short waves use deterministic per-region random phase offsets, blended with continuous first and second derivatives across 24 metre regions. One warped spectral query replaces the old periodic query, without new FFTs. Caustic lookup follows the same phase variation. Mobile and PC both use this variation and local touch behaviour. No external examples or assets are used in this change.

`waterLab.domainTest()` queries real GPU fields: a live local wake remains nonzero, its copy 24 metres away is exactly zero, and an untapered negative control reproduces the old ghost. It also checks regional variation and analytic gradients at region/taper boundaries. Force and mobile interaction tests exercise these checks alongside existing controls. Mobile images intentionally differ from the earlier uniform tiling release.

The optimization regression baseline is now `0bb6aea`, the first local-interaction/regional-variation release. Future optimization passes must again preserve its mobile images and normalized shader operations exactly; the baseline update accepts the requested feature change rather than removing the checks.


## Earth-scale ocean and space flight

The ocean now wraps an analytic sphere with a radius of 6,371,000 metres (diameter 12,742 km). All distances and travel speeds use metres and seconds. This is a procedural ocean planet at Earth's mean spherical size; the project does not contain geographical land or elevation data. Close to the surface, the original FFT water, refraction, caustics, sand and local touch wake remain available. Distant rays transition to the curved ocean, a procedural cloud layer, an approximate exponential atmosphere, sunlight and stars. The same camera and spherical intersection run throughout the transition.

- WASD flies; Q/E descend and rise relative to the current planetary surface. Shift boosts speed.
- With the mouse free, wheel-out smoothly retreats toward space and wheel-in approaches the water. Travel speed grows with altitude.
- Click Fly camera for mouse lock. In this mode, wheel-up increases flight speed and wheel-down reduces it without changing altitude. Escape releases the mouse.
- The Space preset frames the entire globe. Shallows or Reset returns to the surface.
- On touch devices, pinch inward to retreat and spread two fingers to approach. The movement joystick, altitude buttons and screen-drag look remain available. Pinching never pushes the water.

The camera follows great-circle steps and transports its local frame across poles. A single CUDA invocation uses software double arithmetic, stored as high/low float pairs; a small double polynomial rotation avoids the accumulated error of native shader trigonometry. Rendering stays float32 and camera-relative, with altitude separate from the 6,371 km radius. The FFT/material chart remains bounded for close-up precision; it is a local procedural detail layer, not a globally persistent terrain map. There is no planet-sized FFT allocation. A 160-byte telemetry read every 15 frames updates altitude, speed and local weather labels independently of rendering.

`npm run test:planet` checks spherical ray intersections against independent JavaScript double equations, an entire Earth circuit, metre-scale movement on the far side, pole crossing, actual wheel input, flight-speed adjustment, touch pinch input, and real 1440p/4K space framebuffers. It also saves views from orbital height down to the shallow water. The atmosphere and cloud layer are visual approximations. Browser tests use local NVIDIA/Edge and Chromium phone emulation; physical Samsung and Safari behaviour still requires device testing.

The Earth-scale release resets the image/shader optimization baseline to `812431b`. This accepts the requested spherical renderer and preserves its new Mobile/Performance imagery for subsequent optimization work.


## Rotating globe and dynamic weather

Open **Globe weather** to change the simulated clock or season. **Visit a rain system** flies to a rainy cell on the same global map seen from orbit. Pause freezes waves, rain, cloud motion and the clock while leaving the camera usable. **Fixed sunlight study** restores the earlier lighting/wind study for comparisons.

The default clock runs at **120×**, giving one solar day in twelve minutes. Choose real time or another labeled rate. Rain and the wave phase continue at normal simulation speed; wind-energy response follows weather time. Earth-fixed navigation carries the observer with the rotating planet: the Sun and stars rotate in that frame. The model uses a 24-hour mean solar day, a 23.44-degree seasonal tilt approximation, and a separate sidereal star rotation. These follow the distinction in [NASA's reference-systems guide](https://science.nasa.gov/learn/basics-of-space-flight/chapter2-1/) and [Earth fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html). The orbit is approximated as circular.

A deterministic field covers the sphere: trade winds, mid-latitude westerlies, polar easterlies, a seasonally moving tropical wet belt, and six evolving pressure lows with opposite circulation in the two hemispheres. These are simplified patterns motivated by [NOAA's global weather overview](https://www.noaa.gov/education/resource-collections/weather-atmosphere/weather-systems-patterns), not a numerical forecast. It uses no live weather API. Land heating, measured weather observations and an atmospheric fluid solve are not included. Temperature and pressure are indicative simulation values.

Wind is projected into the transported local camera frame and changes the FFT spectrum through per-mode energy memory. The short waves respond sooner than the long waves, with slower decay after the wind subsides. Weather does not regenerate the random spectrum every frame or allocate more FFT cascades. Both hemispheres and the longitude wrap share a continuous world-space field; cache sampling reflects across poles instead of clamping at them.

A bounded 3D cloud volume from 0.8 to 8.2 km uses the same global coverage and density structure as near-water solar attenuation. The reflection cache uses a cheaper two-deck approximation. The local Sun drives caustic transport, submerged lighting, highlights and reflections. Rain has wind-slanted world-space streaks, distance haze and small analytic impact normals. These are visual approximations: cloud lighting uses approximate vertical self-shadowing, rainfall does not solve a separate fluid volume, and local cloud shadowing is represented over the nearby water patch. The bounded FFT/detail chart remains a local approximation rather than a stored globe-wide ocean state.

The global field is cached at 256 × 128 on PC and 128 × 64 on Mobile. The hemispherical sky/reflection cache is 512 × 128 on PC and 256 × 64 on Mobile. A separate view cache integrates the visible cloud volume with 24 PC or 12 mobile samples, at half the water resolution capped at a 1920-pixel long edge on PC or 960 on phones. Premultiplied opacity is reconstructed over the full-resolution ocean and atmosphere. Clouds are filtered with altitude; both caches refresh at 10 Hz or immediately for camera/setting changes. Orbital rendering skips the surface reflection-cache calculation. Shared camera/weather/sky/view storage reserves approximately 66 MiB on desktop or 18 MiB on touch devices, plus 128 KiB of wave-energy memory. Shared storage keeps PC rendering within WebGPU's portable eight-storage-buffer limit. Water detail, FFT dimensions and PC spatial sampling are retained.

Run `npm run test:weather` for solar-period/season checks, both poles and longitude continuity, wind-belt/tangent-vector checks, evolving global conditions, an actual visit to rain, FFT wind coupling with a disabled-weather negative control, pixel-identical pause, pointer-lock success/refusal handling, and real 1440p/4K measurements. `benchmark(samples, true)` measures an advancing scene, including weather and sky-cache work when due; its GPU timestamps still exclude canvas transfer and physical presentation. Phone captures emulate viewport/touch behavior on desktop NVIDIA hardware and do not establish Samsung or Safari GPU performance.

Click **Fly camera** to lock the mouse. The button and status message confirm **Flying**. Wheel-up speeds up and wheel-down slows down in this mode. With the mouse free, the wheel zooms between the sea and space; Ctrl + wheel also zooms while flying. If mouse lock is refused, the UI explains the refusal and enables drag-to-look with WASD and speed scrolling. Escape exits either fly mode.

Distant water uses wind-dependent rough reflection, broad solar glints, and footprint-filtered irregular swell shading. The atmosphere uses 12 PC or 8 mobile integration samples, with denser sampling near the ground in orbital views. Ocean, atmospheric limb, and stars remain at the selected output resolution; the expensive cloud volume uses its separate cache.

The weather release resets the image/shader optimization baseline to `95085db`. It accepts the new moving sunlight, weather response, clouds and distant shading while preserving exact image comparisons for later optimization passes.


## Plate-based globe depths

**Seafloor → Plate-based globe** follows a seeded spherical terrain field as you fly. **Shallows** searches the generated coastline for a daylight shallow patch; **Open water** visits a deep basin. The manual depth slider selects **Manual depth study** for the original optical experiments. **Bathymetry** and **Plates** views show the underlying field without weather obscuring it. Changing **World seed** rebuilds a deterministic world.

Twenty-eight irregular best-candidate sites define spherical Voronoi plates. Each has a seeded Euler rotation and crust classification. Relative tangent velocity distinguishes opening and closing boundaries. A spreading-distance age proxy deepens oceanic crust away from ridges; convergent oceanic boundaries create trenches, while continental crust produces shelves and uplift. The geological ideas follow [USGS's plate-boundary overview](https://pubs.usgs.gov/gip/dynamic/understanding.html). This is synthetic geography and a kinematic landform approximation, not measured Earth bathymetry, an evolved plate reconstruction, or a mantle/erosion simulation. This first depth checkpoint leaves continental crust submerged at a minimum depth of 1.4 m; raised continents follow separately.

CUDA builds a 1024 × 512 depth/crust/boundary cache on desktop, or 512 × 256 on touch devices, only when the seed changes. It costs 8 MiB or 2 MiB, plus a 896-byte plate table and 128 KiB of wave-phase offsets. The same GPU field drives distant shallow-water color and local refraction, light attenuation, caustic depth, sediment response, and finite-depth FFT dispersion. The local optical/FFT patch uses the depth beneath the camera, an approximation over that small patch. A frequency change preserves the accumulated phase instead of making waves jump. Depth labels use the existing asynchronous telemetry read and do not drive the simulation.

`npm run test:geology` checks sites and nearest-plate ownership against an independent CPU oracle, depth distribution, a disabled-tectonics negative control, poles/longitude continuity, seed reproducibility and variation, cache reuse, manual depth, GPU dispersion, phase continuity, and actual 1440p/4K rendering. The older water/weather/planet tests explicitly select the manual-depth study to isolate their existing checks. Geology has its own default-world coverage. Phone checks remain viewport/input emulation on desktop hardware.
