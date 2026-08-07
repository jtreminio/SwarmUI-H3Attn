/** Tuned for 20 sampling steps. Keeps Spectrum's longest run of consecutive forecast
 * steps at 2 — 3-in-a-row is the edge, and 4-5 (what Flex Window 0.75 reaches by 30
 * steps) audibly breaks the generated audio stream.
 *
 * Spectrum's one-point bootstrap (v0.1.7+) is on, which is why Degree is 1 and Warmup
 * Steps is 1 — the bootstrap only exists for that pair, and the node disables it with a
 * console warning under any other. It buys the two forecasts that a degree-4 fit had to
 * spend warming up: a 20-step run goes A F A F ... A A, 9 forecast steps against the 7
 * the old degree-4 / warmup-5 preset reached, with no back-to-back forecasts either way. */
const H3A_OPTIMAL_20 = {
    'H3A Sol Tau': 1.5,
    'H3A Sol Start Percent': 0.2,
    'H3A Sol End Percent': 0.9,
    'H3A Sol Min Tokens': 4096,
    'H3A Sol INT8 QK': true,
    'H3A Sol INT8 PV': true,
    'H3A Sol Sink Conditioning': 'exact_kv_and_rows',
    'H3A Sol Morton': true,
    'H3A Sol Morton Curve': '2d_frame',
    // Dense Blocks and Tau Profile are deliberately absent — which blocks need to stay
    // dense, and which can take a higher tau, is per-model, and the only way to know is a
    // Block Probe run. The preset leaves yours alone.
    'H3A Sol Block Probe': false,
    'H3A Sol Use TMA': false,
    'H3A Sol Verbose': false,
    'H3A Spectrum Blend Weight': 0.5,
    'H3A Spectrum Degree': 1,
    'H3A Spectrum Ridge Lambda': 0.1,
    'H3A Spectrum Window Size': 2.0,
    'H3A Spectrum Flex Window': 0.25,
    'H3A Spectrum Warmup Steps': 1,
    'H3A Spectrum Bootstrap First Forecast': true,
    'H3A Spectrum Tail Actual Steps': 2,
    'H3A Spectrum Max History': 8,
    // History Storage is deliberately absent — the preset leaves whatever you have set.
    'H3A Spectrum Debug': false
};

/** Mirror of SwarmUI's server-side T2IParamTypes.CleanTypeName: lowercase, letters only.
 * "H3A Sol Tau" -> "hasoltau", matching the DOM's "input_hasoltau". */
function h3aParamId(name) {
    return name.toLowerCase().replace(/[^a-z]/g, '');
}

function h3aApplyOptimal20() {
    let applied = 0, missing = [];
    for (let [name, value] of Object.entries(H3A_OPTIMAL_20)) {
        let param = gen_param_types.find(p => p.id == h3aParamId(name));
        if (!param || !document.getElementById(`input_${param.id}`)) {
            missing.push(name);
            continue;
        }
        setDirectParamValue(param, value);
        applied++;
    }
    // Setting a value auto-activates its group chain, but be explicit rather than rely on that.
    for (let groupId of ['hsolattn', 'hspectrum', 'hattention']) {
        let toggle = document.getElementById(`input_group_content_${groupId}_toggle`);
        if (toggle && !toggle.checked) {
            toggle.checked = true;
            doToggleGroup(`input_group_content_${groupId}`);
        }
    }
    if (missing.length) {
        console.log(`[H3Attn] Optimal @ 20: applied ${applied}, skipped ${missing.length} (nodes not installed?): ${missing.join(', ')}`);
    }
}

postParamBuildSteps.push(() => {
    let targetGroup = document.getElementById('input_group_content_hattention');
    if (targetGroup && !document.getElementById('h3attn_optimal20_button')) {
        targetGroup.prepend(createDiv('h3attn_optimal20_button', 'keep_group_visible',
            `<button class="basic-button" onclick="h3aApplyOptimal20()" title="Set every Sol-Attn and Spectrum value to the tuned 20-step preset. Does not change your Steps parameter.">Optimal @ 20 Steps</button>`));
    }
});
