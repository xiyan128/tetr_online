//! CRT / retro-arcade post-process pass.
//!
//! A single fullscreen render-graph node runs after tonemapping on every
//! [`Camera2d`] and re-samples the frame through [`crt.wgsl`](../../assets/shaders/crt.wgsl):
//! curvature, chromatic aberration, scanlines, an aperture-grille mask, vignette,
//! and a faint flicker. Applying it globally (menus included) keeps the whole game
//! behind one pane of glass.
//!
//! Mechanism (the verified WebGL2-safe path, adapted from Bevy's official
//! `custom_post_processing` example to 2D): a [`ViewNode`] that, per view, grabs
//! the [`ViewTarget`]'s double-buffered `post_process_write()` and draws a
//! fullscreen triangle with our fragment shader. The effect parameters travel as a
//! per-camera [`CrtSettings`] component, extracted to the render world and uploaded
//! as a uniform automatically. That uniform is two `vec4`s — 16-byte aligned by
//! construction, so it is valid under WebGL2's std140 layout without the
//! conditional-padding dance the upstream example needs.
//!
//! Runs on both the default WebGL2 web bundle and the WebGPU bundle; it touches
//! only the composited LDR image, so it is independent of HDR / bloom.

use bevy::{
    core_pipeline::{
        core_2d::graph::{Core2d, Node2d},
        FullscreenShader,
    },
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{sampler, texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewTarget,
        RenderApp, RenderStartup,
    },
    ui_render::graph::NodeUi,
};

const SHADER_ASSET_PATH: &str = "shaders/crt.wgsl";

/// Tunable CRT look, shared by every camera. Edit these (live, via the dev
/// inspector) to dial the effect in. All values are gentle by default so menu text
/// stays readable.
#[derive(Resource, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct CrtConfig {
    /// Barrel-distortion strength (screen bulge). `0` = flat.
    pub curvature: f32,
    /// How dark the scanline troughs get, `0..1`.
    pub scanline_intensity: f32,
    /// Radial red/blue separation at the edges (UV units).
    pub aberration: f32,
    /// Corner darkening, `0..1`.
    pub vignette: f32,
    /// Aperture-grille tint depth, `0..1`. Subtle by default to avoid moiré.
    pub mask_intensity: f32,
    /// Overall brightness lift, compensating the darkening from scanlines/vignette.
    pub brightness: f32,
}

impl Default for CrtConfig {
    fn default() -> Self {
        Self {
            // Gentle curvature: enough to read as CRT glass without warping the UI
            // away from where pointer clicks land (there's no overscan zoom to mask
            // a stronger bulge — see `curve_uv`).
            curvature: 0.04,
            scanline_intensity: 0.13,
            aberration: 0.003,
            vignette: 0.32,
            mask_intensity: 0.07,
            brightness: 1.15,
        }
    }
}

impl CrtConfig {
    /// Pack the knobs (+ the current time and the on/off flag) into the GPU uniform
    /// layout. `enabled` rides in `params_b.w`; when off the shader passes the frame
    /// through untouched (the node still runs, but as a no-op blit).
    fn to_settings(self, time: f32, enabled: bool) -> CrtSettings {
        CrtSettings {
            params_a: Vec4::new(time, self.curvature, self.scanline_intensity, self.aberration),
            params_b: Vec4::new(
                self.vignette,
                self.mask_intensity,
                self.brightness,
                enabled as u32 as f32,
            ),
        }
    }
}

/// Per-camera GPU uniform for the CRT pass. Two `vec4`s == 32 bytes, 16-byte
/// aligned, so it is valid std140 on WebGL2 with no conditional padding. Carried
/// on the camera entity, extracted to the render world, and uploaded each frame.
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct CrtSettings {
    params_a: Vec4,
    params_b: Vec4,
}

/// CRT post-processing for all 2D cameras.
pub struct CrtPlugin;

impl Plugin for CrtPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CrtConfig>()
            .register_type::<CrtConfig>()
            .add_plugins((
                ExtractComponentPlugin::<CrtSettings>::default(),
                UniformComponentPlugin::<CrtSettings>::default(),
            ))
            // Keep every 2D camera's CRT uniform attached and current.
            .add_systems(Update, sync_crt_settings);

        // Render-world wiring: build the pipeline once, then slot the node into the
        // 2D graph between tonemapping and the end of post-processing.
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_crt_pipeline)
            .add_render_graph_node::<ViewNodeRunner<CrtNode>>(Core2d, CrtLabel)
            .add_render_graph_edges(
                Core2d,
                // Run AFTER the UI pass — which composites the HUD/menus on top of the
                // post-process chain — so the CRT warp covers the UI too, and BEFORE
                // the final upscaling blit (so we still write into the view's own
                // ping-pong texture).
                (NodeUi::UiPass, CrtLabel, Node2d::Upscaling),
            );
    }
}

/// Ensure every `Camera2d` carries an up-to-date [`CrtSettings`]. Attaches the
/// component on first sight and refreshes the animated `time` each frame, so the
/// effect covers menus and gameplay alike with no per-camera bookkeeping.
///
/// The unconditional per-frame write is intentional: `time` drives the scanline
/// roll and flicker, so the uniform genuinely changes every frame — there is
/// nothing to change-guard here (unlike `sync_bloom_toggle`, whose value is steady).
fn sync_crt_settings(
    mut commands: Commands,
    config: Res<CrtConfig>,
    toggles: Res<crate::vfx::VfxToggles>,
    time: Res<Time>,
    mut cameras: Query<(Entity, Option<&mut CrtSettings>), With<Camera2d>>,
) {
    let settings = config.to_settings(time.elapsed_secs(), toggles.crt);
    for (entity, existing) in &mut cameras {
        match existing {
            Some(mut current) => *current = settings,
            None => {
                commands.entity(entity).insert(settings);
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct CrtLabel;

#[derive(Default)]
struct CrtNode;

impl ViewNode for CrtNode {
    type ViewQuery = (
        &'static ViewTarget,
        // Only run on cameras that carry the settings (all of them, in practice).
        &'static CrtSettings,
        &'static DynamicUniformIndex<CrtSettings>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, _settings, settings_index): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let crt_pipeline = world.resource::<CrtPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // Pick the pipeline whose target format matches this view: an HDR (bloom)
        // gameplay view needs the HDR pipeline; menus and the WebGL2 build need LDR.
        let pipeline_id = crt_pipeline.pipeline_for(view_target.main_texture_format());
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
            // Shader still loading/compiling — pass the frame through untouched.
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<CrtSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        // Flip the view target's double buffer: read `source`, write `destination`.
        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "crt_bind_group",
            &pipeline_cache.get_bind_group_layout(&crt_pipeline.layout),
            &BindGroupEntries::sequential((
                post_process.source,
                &crt_pipeline.sampler,
                settings_binding.clone(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("crt_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[settings_index.index()]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

/// Global render data for the CRT pass, built once on render startup.
#[derive(Resource)]
struct CrtPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    /// Pipeline for the standard LDR view format — menus, and the gameplay view
    /// whenever bloom is off (always, on the WebGL2 bundle).
    pipeline_ldr: CachedRenderPipelineId,
    /// Pipeline for the HDR view format, used when the gameplay camera runs bloom.
    /// Only built when the `bloom` feature can produce HDR views.
    #[cfg(feature = "bloom")]
    pipeline_hdr: CachedRenderPipelineId,
}

impl CrtPipeline {
    /// The pipeline whose color-target format matches `format`. The CRT pass writes
    /// back into the view's own texture, so the formats must agree.
    fn pipeline_for(&self, format: TextureFormat) -> CachedRenderPipelineId {
        #[cfg(feature = "bloom")]
        if format == ViewTarget::TEXTURE_FORMAT_HDR {
            return self.pipeline_hdr;
        }
        let _ = format;
        self.pipeline_ldr
    }
}

fn init_crt_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "crt_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // Screen texture + a filtering sampler (linear, so curvature and
                // aberration resample smoothly rather than blockily).
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<CrtSettings>(true),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("crt_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    let shader = asset_server.load(SHADER_ASSET_PATH);
    // One pipeline per possible view-target format (the only difference is the
    // fragment color-target format); the node selects the right one per view.
    let queue = |format: TextureFormat| {
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("crt_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        })
    };

    let pipeline_ldr = queue(TextureFormat::bevy_default());
    #[cfg(feature = "bloom")]
    let pipeline_hdr = queue(ViewTarget::TEXTURE_FORMAT_HDR);

    commands.insert_resource(CrtPipeline {
        layout,
        sampler,
        pipeline_ldr,
        #[cfg(feature = "bloom")]
        pipeline_hdr,
    });
}
