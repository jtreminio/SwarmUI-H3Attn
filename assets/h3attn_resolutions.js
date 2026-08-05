/** 32x-multiple resolution browser for MiniMax H3.
 *
 * Every listed size is an exact ratio match with both axes a multiple of 32
 * (SwarmUI's CANVAS_MULTIPLE). The useful extra column is tokens-per-frame:
 * H3's VAE is /16 and the DiT patches 2x2, so a frame is (W/32)*(H/32) tokens.
 * When that divides evenly by the Sol-Attn kernel's 64-token block, the
 * 2d_frame Morton curve's within-frame tiles line up with the routing blocks;
 * when it does not, they drift and the video can ripple. */

const H3A_RATIOS = [[1, 1], [4, 3], [3, 2], [8, 5], [16, 9], [21, 9], [3, 4], [2, 3], [5, 8], [9, 16], [9, 21]];
const H3A_BLOCK = 64;
const H3A_MAX_PIXELS = 768 * 1344;   // MiniMaxH3 node's MAX_PIXELS
const H3A_MIN_AREA = 400000;         // below this is not worth listing
const H3A_MAX_AREA = H3A_MAX_PIXELS * 1.35;

function h3aGcd(a, b) {
    return b ? h3aGcd(b, a % b) : a;
}

function h3aResolutionsFor(a, b) {
    let g = h3aGcd(a, b), ra = a / g, rb = b / g, out = [];
    for (let k = 1; ; k++) {
        let w = 32 * ra * k, h = 32 * rb * k, area = w * h;
        if (area > H3A_MAX_AREA) {
            break;
        }
        if (area >= H3A_MIN_AREA) {
            let tokens = (w / 32) * (h / 32);
            out.push({ w: w, h: h, area: area, tokens: tokens, aligned: tokens % H3A_BLOCK == 0, overCap: area > H3A_MAX_PIXELS });
        }
    }
    return out;
}

function h3aApplyResolution(w, h) {
    for (let [id, value] of [['width', w], ['height', h]]) {
        let param = gen_param_types.find(p => p.id == id);
        if (param && document.getElementById(`input_${param.id}`)) {
            setDirectParamValue(param, value);
        }
        else {
            console.log(`[H3Attn] cannot set ${id}: parameter not present`);
        }
    }
    $('#h3attn_resolutions_modal').modal('hide');
}

function h3aBuildResolutionsModal() {
    let body = '';
    for (let [a, b] of H3A_RATIOS) {
        let rows = h3aResolutionsFor(a, b);
        body += `<h6 style="margin-top: 0.9rem;">${a}:${b}</h6>`;
        if (!rows.length) {
            body += `<div class="translate">No 32-multiple size in range.</div>`;
            continue;
        }
        body += `<table class="table table-sm" style="margin-bottom: 0;"><thead><tr>
            <th>Resolution</th><th>MP</th><th>Tokens/frame</th><th>64-blocks</th><th></th></tr></thead><tbody>`;
        for (let r of rows) {
            let blocks = r.aligned ? `${r.tokens / H3A_BLOCK} &check;` : (r.tokens / H3A_BLOCK).toFixed(2);
            let note = r.overCap ? '<span title="Above the MiniMax H3 node\'s 768x1344 pixel cap">over cap</span>' : '';
            let emphasis = r.aligned ? ' style="font-weight: bold;"' : '';
            body += `<tr${emphasis}>
                <td><a href="#" onclick="h3aApplyResolution(${r.w}, ${r.h}); return false;">${r.w} &times; ${r.h}</a></td>
                <td>${(r.area / 1000000).toFixed(2)}</td>
                <td>${r.tokens}</td>
                <td>${blocks}</td>
                <td>${note}</td></tr>`;
        }
        body += `</tbody></table>`;
    }
    let html = `
    <div class="modal" tabindex="-1" role="dialog" id="h3attn_resolutions_modal">
        <div class="modal-dialog modal-lg modal-dialog-scrollable" role="document">
            <div class="modal-content">
                <div class="modal-header"><h5 class="modal-title translate">32x Resolutions</h5></div>
                <div class="modal-body">
                    <div class="translate">Exact-ratio sizes with both axes a multiple of 32. Bold rows divide evenly into the Sol-Attn kernel's 64-token blocks, which is what the 2d_frame Morton curve needs to stay aligned. Click a resolution to apply it.</div>
                    ${body}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary translate" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>`;
    let holder = createDiv('h3attn_resolutions_modal_holder', null, html);
    document.body.appendChild(holder);
}

function h3aShowResolutions() {
    if (!document.getElementById('h3attn_resolutions_modal')) {
        h3aBuildResolutionsModal();
    }
    $('#h3attn_resolutions_modal').modal('show');
}

postParamBuildSteps.push(() => {
    let targetGroup = document.getElementById('input_group_content_hattention');
    if (targetGroup && !document.getElementById('h3attn_resolutions_button')) {
        targetGroup.prepend(createDiv('h3attn_resolutions_button', 'keep_group_visible',
            `<button class="basic-button" onclick="h3aShowResolutions()" title="Browse exact-ratio resolutions with both axes a multiple of 32, showing which ones align to the Sol-Attn 64-token block.">32x Resolutions</button>`));
    }
});
