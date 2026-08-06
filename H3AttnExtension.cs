using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace H3Attn;

/// <summary>Attaches the Sol-Attn and Spectrum MiniMax H3 accelerator nodes onto every
/// <c>MiniMaxH3SigmaShift</c> node in the generated workflow.</summary>
public class H3AttnExtension : Extension
{
    /// <summary>Core node this extension hangs off of — emitted by WorkflowGeneratorModelSupport for every H3 model load.</summary>
    public const string SigmaShiftNodeName = "MiniMaxH3SigmaShift";

    public const string SolFeatureFlag = "sol_attn_triton";
    public const string SolNodeName = "SolAttnPatch";
    public const string SolProbeNodeName = "SolAttnBlockProbe";

    public const string SpectrumFeatureFlag = "spectrum_minimax_h3";
    public const string SpectrumNodeName = "SpectrumApplyMiniMaxH3";

    /// <summary>Runs after every core workflow step (the last core step is priority 200).</summary>
    public const double StepPriority = 1000;

    public static T2IParamGroup H3AttnGroup, SolGroup, SpectrumGroup;

    // Sol-Attn (gate: SolTau — see IsSubGroupEnabled).
    public static T2IRegisteredParam<double> SolTau, SolStartPercent, SolEndPercent;
    public static T2IRegisteredParam<int> SolMinTokens;
    public static T2IRegisteredParam<bool> SolInt8Qk, SolInt8Pv, SolMorton, SolUseTma, SolBlockProbe, SolVerbose;
    public static T2IRegisteredParam<string> SolSinkConditioning, SolMortonCurve, SolDenseBlocks;

    // Spectrum (gate: SpectrumBlendWeight).
    public static T2IRegisteredParam<double> SpectrumBlendWeight, SpectrumRidgeLambda, SpectrumWindowSize, SpectrumFlexWindow;
    public static T2IRegisteredParam<int> SpectrumDegree, SpectrumWarmupSteps, SpectrumTailActualSteps, SpectrumMaxHistory;
    public static T2IRegisteredParam<bool> SpectrumDebug;
    public static T2IRegisteredParam<string> SpectrumHistoryStorage;

    public override void OnInit()
    {
        Logs.Info("SwarmUI H3 Attn Extension initializing...");
        ComfyUIBackendExtension.NodeToFeatureMap[SolNodeName] = SolFeatureFlag;
        ComfyUIBackendExtension.NodeToFeatureMap[SolProbeNodeName] = SolFeatureFlag;
        ComfyUIBackendExtension.NodeToFeatureMap[SpectrumNodeName] = SpectrumFeatureFlag;
        InstallableFeatures.RegisterInstallableFeature(new(
            "Sol-Attn (Triton)",
            SolFeatureFlag,
            "https://github.com/kijai/ComfyUI-SolAttn_triton",
            "kijai"
        ));
        InstallableFeatures.RegisterInstallableFeature(new(
            "Spectrum MiniMax H3",
            SpectrumFeatureFlag,
            "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3",
            "xmarre"
        ));
        ScriptFiles.Add("assets/h3attn_install.js");
        ScriptFiles.Add("assets/h3attn_preset.js");
        ScriptFiles.Add("assets/h3attn_resolutions.js");

        H3AttnGroup = new T2IParamGroup(
            Name: "H3 Attention",
            Toggles: true,
            Open: false,
            IsAdvanced: false,
            OrderPriority: 9,
            Description: "MiniMax H3 sampling accelerators. Attached to every MiniMaxH3SigmaShift node in the workflow, in order: Sol-Attn, then Spectrum."
        );

        SolGroup = new T2IParamGroup(
            Name: "H3 Sol-Attn",
            Toggles: true,
            Open: false,
            IsAdvanced: false,
            OrderPriority: 1,
            Parent: H3AttnGroup,
            Description: "Sparse self-attention (arXiv 2607.24027) via Triton. bf16 + head_dim 128 only; anything else falls back to the normal attention backend. First run is slow while Triton autotunes."
        );

        SpectrumGroup = new T2IParamGroup(
            Name: "H3 Spectrum",
            Toggles: true,
            Open: false,
            IsAdvanced: false,
            OrderPriority: 2,
            Parent: H3AttnGroup,
            Description: "Chebyshev-ridge spectral forecasting of post-transformer features, so some solver steps skip the transformer entirely. Approximate: it changes the denoising trajectory."
        );

        int order = 0;

        SolTau = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Sol Tau",
            Description: "Sparsity threshold beta. Higher is sparser and faster: 1.0 keeps ~16% of blocks exact, 1.5 ~7%, 2.0 ~2.7%.",
            Default: "1.30",
            Min: 0, Max: 4, Step: 0.05,
            ViewMin: 0, ViewMax: 4,
            ViewType: ParamViewType.SLIDER,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolStartPercent = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Sol Start Percent",
            Description: "Run dense attention before this point in the sampling schedule. The paper uses 0.2.",
            Default: "0.20",
            Min: 0, Max: 1, Step: 0.01,
            ViewMin: 0, ViewMax: 1,
            ViewType: ParamViewType.SLIDER,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolEndPercent = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Sol End Percent",
            Description: "Run dense attention after this point in the sampling schedule.",
            Default: "0.90",
            Min: 0, Max: 1, Step: 0.01,
            ViewMin: 0, ViewMax: 1,
            ViewType: ParamViewType.SLIDER,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolMinTokens = T2IParamTypes.Register<int>(new T2IParamType(
            Name: "H3A Sol Min Tokens",
            Description: "Sequences shorter than this stay dense. Short clips are already cheap, so sparsifying them only costs accuracy.",
            Default: "4096",
            Min: 0, Max: 1048576, Step: 512,
            IsAdvanced: true,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolInt8Qk = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Sol INT8 QK",
            Description: "INT8 QK in the exact branch (Sage-style smoothed K with per-token scales). Helps at tau <= 1.5, a net loss at tau >= 2.0 where the quantize pass outweighs the shrinking exact branch.",
            Default: "true",
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolInt8Pv = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Sol INT8 PV",
            Description: "Also run the exact branch's P@V in INT8 (per-row P scale, per-channel V scale). PV and QK cost the same, so this is the other half of the INT8 win. Only applies when INT8 QK is on.",
            Default: "true",
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolSinkConditioning = T2IParamTypes.Register<string>(new T2IParamType(
            Name: "H3A Sol Sink Conditioning",
            Description: "exact_kv: every query sees the packed text/audio/reference rows exactly (~3% cost). exact_kv_and_rows: also runs those query rows dense, making the generated audio stream exact (~20% cost). off: no sink.",
            Default: "exact_kv_and_rows",
            GetValues: _ => ["exact_kv", "exact_kv_and_rows", "off"],
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolMorton = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Sol Morton",
            Description: "Reorder video tokens into Morton (Z-order) so each 64-token block is a compact 3D neighbourhood instead of a 2-row strip, which makes routing far more accurate at a given density.",
            Default: "false",
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolMortonCurve = T2IParamTypes.Register<string>(new T2IParamType(
            Name: "H3A Sol Morton Curve",
            Description: "2d_frame Z-orders within each frame and leaves frame order alone — faster, but it can warp the video (a rippling, water-drop distortion) when a frame's token count is not a multiple of the kernel's 64-token block size, since the within-frame tiles then drift out of alignment with the routing blocks. Use the 32x Resolutions browser to pick a resolution where tokens-per-frame divides by 64. 3d interleaves t/h/w equally so each routing block is a compact space-time volume, which is immune to that drift.",
            Default: "2d_frame",
            GetValues: _ => ["3d", "2d_frame"],
            IsAdvanced: true,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolDenseBlocks = T2IParamTypes.Register<string>(new T2IParamType(
            Name: "H3A Sol Dense Blocks",
            Description: "Transformer blocks to keep dense, e.g. '0-2,-1' for the first three and the last. Negative indices count from the end. The first and last blocks are the most approximation-sensitive — their error reaches the output with no later block to absorb it. Empty sparsifies every block. Use Block Probe to find which ones actually matter for your model.",
            Default: "",
            Examples: ["", "0,-1", "0-2,-1"],
            IsAdvanced: true,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolBlockProbe = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Sol Block Probe",
            Description: "Diagnostic. Runs every attention call both sparse and dense and logs each block's relative error, worst first, when sampling ends — paste the top entries into Dense Blocks. The generation itself is the dense reference, and it costs roughly dense + sparse, so turn this back off once you have the numbers.",
            Default: "false",
            IsAdvanced: true,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolUseTma = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Sol Use TMA",
            Description: "Use the TMA descriptor kernels instead of the pointer ones. Descriptors address strided inputs directly, so peak VRAM now matches the pointer path. Requires SM90+ and Triton 3.3+; has not measured faster on any tested GPU.",
            Default: "false",
            IsAdvanced: true,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        SolVerbose = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Sol Verbose",
            Description: "Log Sol-Attn kernel path and autotune details to the ComfyUI console.",
            Default: "false",
            IsAdvanced: true,
            Group: SolGroup,
            FeatureFlag: SolFeatureFlag,
            OrderPriority: order++
        ));

        order = 0;

        SpectrumBlendWeight = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Spectrum Blend Weight",
            Description: "How much of the forecast is blended into the hidden features on forecast steps. Lower is closer to the native trajectory.",
            Default: "0.50",
            Min: 0, Max: 1, Step: 0.01,
            ViewMin: 0, ViewMax: 1,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumDegree = T2IParamTypes.Register<int>(new T2IParamType(
            Name: "H3A Spectrum Degree",
            Description: "Chebyshev polynomial degree used to fit the feature history.",
            Default: "4",
            Min: 1, Max: 16, Step: 1,
            ViewMin: 1, ViewMax: 16,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumRidgeLambda = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Spectrum Ridge Lambda",
            Description: "Ridge regularization on the Chebyshev fit. Higher damps the extrapolation.",
            Default: "0.10",
            Min: 0, Max: 10, Step: 0.01,
            ViewMin: 0, ViewMax: 10,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumWindowSize = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Spectrum Window Size",
            Description: "Size of the fitting window over recent actual steps.",
            Default: "2.00",
            Min: 1, Max: 16, Step: 0.05,
            ViewMin: 1, ViewMax: 16,
            ViewType: ParamViewType.SLIDER,
            IsAdvanced: true,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumFlexWindow = T2IParamTypes.Register<double>(new T2IParamType(
            Name: "H3A Spectrum Flex Window",
            Description: "Extra slack the fitting window is allowed to stretch by.",
            Default: "0.75",
            Min: 0, Max: 8, Step: 0.05,
            ViewMin: 0, ViewMax: 8,
            ViewType: ParamViewType.SLIDER,
            IsAdvanced: true,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumWarmupSteps = T2IParamTypes.Register<int>(new T2IParamType(
            Name: "H3A Spectrum Warmup Steps",
            Description: "Leading steps that always run the real transformer, to build feature history before any forecasting.",
            Default: "5",
            Min: 0, Max: 64, Step: 1,
            ViewMin: 0, ViewMax: 64,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumTailActualSteps = T2IParamTypes.Register<int>(new T2IParamType(
            Name: "H3A Spectrum Tail Actual Steps",
            Description: "Trailing steps that always run the real transformer, so the final detail pass is never forecast.",
            Default: "1",
            Min: 0, Max: 64, Step: 1,
            ViewMin: 0, ViewMax: 64,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumMaxHistory = T2IParamTypes.Register<int>(new T2IParamType(
            Name: "H3A Spectrum Max History",
            Description: "How many past actual-step feature tensors to retain for fitting.",
            Default: "8",
            Min: 2, Max: 64, Step: 1,
            ViewMin: 2, ViewMax: 64,
            ViewType: ParamViewType.SLIDER,
            IsAdvanced: true,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumHistoryStorage = T2IParamTypes.Register<string>(new T2IParamType(
            Name: "H3A Spectrum History Storage",
            Description: "Where the feature history lives. vram is faster but the H3 packed hidden state is large at long durations.",
            Default: "system_ram",
            GetValues: _ => ["system_ram", "vram"],
            IsAdvanced: true,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        SpectrumDebug = T2IParamTypes.Register<bool>(new T2IParamType(
            Name: "H3A Spectrum Debug",
            Description: "Log per-step forecast/actual decisions to the ComfyUI console.",
            Default: "false",
            IsAdvanced: true,
            Group: SpectrumGroup,
            FeatureFlag: SpectrumFeatureFlag,
            OrderPriority: order++
        ));

        WorkflowGenerator.AddStep(ApplyH3Attn, StepPriority);
    }

    /// <summary>A toggled param group sends none of its params when its toggle (or any parent's) is off,
    /// so the presence of a group's gate param is exactly "this sub-group is enabled".</summary>
    private static bool IsSubGroupEnabled<T>(WorkflowGenerator generator, T2IRegisteredParam<T> gateParam)
        => generator.UserInput.TryGet(gateParam, out _);

    private static void ApplyH3Attn(WorkflowGenerator generator)
    {
        bool useSol = IsSubGroupEnabled(generator, SolTau);
        bool useSpectrum = IsSubGroupEnabled(generator, SpectrumBlendWeight);
        bool useProbe = useSol && generator.UserInput.Get(SolBlockProbe, false);
        if (!useSol && !useSpectrum)
        {
            return;
        }
        List<string> shiftIds = [];
        generator.RunOnNodesOfClass(SigmaShiftNodeName, (id, _) => shiftIds.Add(id));
        if (shiftIds.Count == 0)
        {
            Logs.Debug($"H3 Attn: enabled but the workflow has no {SigmaShiftNodeName} node (not a MiniMax H3 generation); skipping.");
            return;
        }
        foreach (string shiftId in shiftIds)
        {
            JArray shiftOut = new(shiftId, 0);
            string headId = null, tailId = null;
            if (useSol)
            {
                headId = tailId = CreateSolAttnNode(generator);
                if (useProbe)
                {
                    // The probe wraps whatever override is already installed, so it has to sit after the patch.
                    JObject probeInputs = new() { ["model"] = new JArray(tailId, 0) };
                    tailId = generator.CreateNode(SolProbeNodeName, (_, node) => node["inputs"] = probeInputs);
                }
            }
            if (useSpectrum)
            {
                tailId = CreateSpectrumNode(generator, tailId is null ? null : new JArray(tailId, 0));
                headId ??= tailId;
            }
            // Repoint every existing consumer of the sigma-shift output at the tail of the new chain,
            // then feed the head from the sigma shift. Order matters: doing it the other way round
            // would rewrite the head's own model input into a self-reference.
            generator.ReplaceNodeConnection(shiftOut, new JArray(tailId, 0));
            ((JObject)generator.Workflow[headId]["inputs"])["model"] = shiftOut;
        }
        Logs.Debug($"H3 Attn: attached{(useSol ? " Sol-Attn" : "")}{(useProbe ? " Block-Probe" : "")}{(useSpectrum ? " Spectrum" : "")} to {shiftIds.Count} {SigmaShiftNodeName} node(s).");
    }

    /// <summary>Creates the Sol-Attn patch node with no model wired yet (the caller connects it).</summary>
    private static string CreateSolAttnNode(WorkflowGenerator generator)
    {
        JObject inputs = new()
        {
            ["tau"] = generator.UserInput.Get(SolTau, 1.3),
            ["start_percent"] = generator.UserInput.Get(SolStartPercent, 0.2),
            ["end_percent"] = generator.UserInput.Get(SolEndPercent, 0.9),
            ["min_tokens"] = generator.UserInput.Get(SolMinTokens, 4096),
            ["int8_qk"] = generator.UserInput.Get(SolInt8Qk, true),
            ["int8_pv"] = generator.UserInput.Get(SolInt8Pv, true),
            ["sink_conditioning"] = generator.UserInput.Get(SolSinkConditioning, "exact_kv_and_rows"),
            ["morton"] = generator.UserInput.Get(SolMorton, false),
            ["morton_curve"] = generator.UserInput.Get(SolMortonCurve, "2d_frame"),
            ["dense_blocks"] = generator.UserInput.Get(SolDenseBlocks, ""),
            ["verbose"] = generator.UserInput.Get(SolVerbose, false),
            ["use_tma"] = generator.UserInput.Get(SolUseTma, false)
        };
        // The configure-action overload of CreateNode skips the input-hash dedupe cache, so two
        // sigma-shift nodes each get their own patch node instead of collapsing into one.
        return generator.CreateNode(SolNodeName, (_, node) => node["inputs"] = inputs);
    }

    private static string CreateSpectrumNode(WorkflowGenerator generator, JArray model)
    {
        JObject inputs = new()
        {
            ["enabled"] = true,
            ["blend_weight"] = generator.UserInput.Get(SpectrumBlendWeight, 0.5),
            ["degree"] = generator.UserInput.Get(SpectrumDegree, 4),
            ["ridge_lambda"] = generator.UserInput.Get(SpectrumRidgeLambda, 0.1),
            ["window_size"] = generator.UserInput.Get(SpectrumWindowSize, 2.0),
            ["flex_window"] = generator.UserInput.Get(SpectrumFlexWindow, 0.75),
            ["warmup_steps"] = generator.UserInput.Get(SpectrumWarmupSteps, 5),
            ["tail_actual_steps"] = generator.UserInput.Get(SpectrumTailActualSteps, 1),
            ["max_history"] = generator.UserInput.Get(SpectrumMaxHistory, 8),
            ["debug"] = generator.UserInput.Get(SpectrumDebug, false),
            ["history_storage"] = generator.UserInput.Get(SpectrumHistoryStorage, "system_ram")
        };
        if (model is not null)
        {
            inputs["model"] = model;
        }
        return generator.CreateNode(SpectrumNodeName, (_, node) => node["inputs"] = inputs);
    }
}
