import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

import gradio as gr

from soulx_svc import SVCRunParams, SVCServiceRunner


ROOT = Path(__file__).parent
PROMPT_MAX_SEC_DEFAULT = 30
TARGET_MAX_SEC_DEFAULT = 600

SVC_EXAMPLE_PROMPT_AUDIO = "example/audio/svc_prompt_demo.mp3"
SVC_EXAMPLE_TARGET_AUDIO = "example/audio/svc_target_demo.mp3"

EXAMPLE_LIST = [[
	str(ROOT / SVC_EXAMPLE_PROMPT_AUDIO),
	str(ROOT / SVC_EXAMPLE_TARGET_AUDIO),
	False,
	True,
	True,
	True,
	0,
	32,
	1.0,
	42,
]]

_I18N = dict(
	display_lang_label=dict(en="Display Language", zh="显示语言"),
	title=dict(en="## SoulX-Singer SVC", zh="## SoulX-Singer SVC"),
	prompt_audio_label=dict(en=f"Prompt audio", zh=f"Prompt 音频"),
	target_audio_label=dict(en=f"Target audio", zh=f"Target 音频"),
	prompt_vocal_sep_label=dict(en="Prompt vocal separation", zh="Prompt 人声分离"),
	target_vocal_sep_label=dict(en="Target vocal separation", zh="Target 人声分离"),
	auto_shift_label=dict(en="Auto pitch shift", zh="自动变调"),
	auto_mix_acc_label=dict(en="Auto mix accompaniment", zh="自动混合伴奏"),
	pitch_shift_label=dict(en="Pitch shift (semitones)", zh="指定变调（半音）"),
	n_step_label=dict(en="n_step", zh="采样步数"),
	cfg_label=dict(en="cfg scale", zh="cfg系数"),
	seed_label=dict(en="Seed", zh="种子"),
	examples_label=dict(en="Reference example (click to load)", zh="参考样例（点击加载）"),
	run_btn=dict(en="🎤Singing Voice Conversion", zh="🎤歌声转换"),
	output_audio_label=dict(en="Generated audio", zh="合成结果音频"),
	warn_missing_audio=dict(en="Please provide both prompt audio and target audio.", zh="请同时上传 Prompt 与 Target 音频。"),
	instruction_title=dict(en="Usage", zh="使用说明"),
	instruction_p1=dict(
        en="Upload the Prompt and Target audio, and configure the parameters",
        zh="上传 Prompt 与 Target 音频，并配置相关参数",
    ),
    instruction_p2=dict(
        en="Click「🎤Singing Voice Conversion」to start singing voice conversion.",
        zh="点击「🎤歌声转换」开始最终生成。",
    ),
	tips_title=dict(en="Tips", zh="提示"),
	tip_p1=dict(
        en="Input: The Prompt audio is recommended to be a clean and clear singing voice, while the Target audio can be either a pure vocal or a mixture with accompaniment. If the audio contains accompaniment, please check the vocal separation option.",
        zh="输入：Prompt 音频建议是干净清晰的歌声，Target 音频可以是纯歌声或伴奏，这两者若带伴奏需要勾选分离选项",
    ),
	tip_p2=dict(
        en="Pitch shift: When there is a large pitch range difference between the Prompt and Target audio, you can try enabling auto pitch shift or manually adjusting the pitch shift in semitones. When a non-zero pitch shift is specified, auto pitch shift will not take effect. The accompaniment of auto mix will be pitch-shifted together with the vocal (keeping the same octave).",
        zh="变调：Prompt 音频的音域和 Target 音频的音域差距较大的时候，可以尝试开启自动变调或手动调整变调半音数，指定非0的变调半音数时，自动变调不生效，自动混音的伴奏会配合歌声进行升降调（保持同一个八度）",
    ),
	tip_p3=dict(
        en="Model parameters: Generally, a larger number of sampling steps will yield better generation quality but also longer generation time; a larger cfg scale will increase timbre similarity and melody fidelity, but may cause more distortion, it is recommended to take a value between 1 and 3.",
        zh="模型参数：一般采样步数越大，生成质量越好，但生成时间也越长；一般cfg系数越大，音色相似度和旋律保真度越高，但是会造成更多的失真，建议取1～3之间的值",
    ),
	tip_p4=dict(
        en="If you want to convert a long audio or a whole song with large pitch range, there may be instability in the generated voice. You can try converting in segments.",
        zh="长音频或完整歌曲中，音域变化较大的情况有可能出现音色不稳定，可以尝试分段转换",
    )
)

_GLOBAL_LANG: Literal["zh", "en"] = "zh"


def _i18n(key: str) -> str:
	return _I18N[key][_GLOBAL_LANG]


def _print_exception(context: str) -> None:
	print(f"[{context}]\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _session_dir() -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return ROOT / "outputs" / "gradio" / "svc" / timestamp


def _normalize_audio_input(audio):
	return audio[0] if isinstance(audio, tuple) else audio


def _usage_md() -> str:
	return "\n\n".join([
		f"### {_i18n('instruction_title')}",
		f"**1.** {_i18n('instruction_p1')}",
		f"**2.** {_i18n('instruction_p2')}",
	])


def _tips_md() -> str:
	return "\n\n".join([
		f"### {_i18n('tips_title')}",
		f"- {_i18n('tip_p1')}",
		f"- {_i18n('tip_p2')}",
		f"- {_i18n('tip_p3')}",
		f"- {_i18n('tip_p4')}",
	])


APP_STATE = SVCServiceRunner(repo_root=ROOT, use_fp16="--fp16" in sys.argv)


def _start_svc(prompt_audio, target_audio, prompt_vocal_sep, target_vocal_sep, auto_shift, auto_mix_acc, pitch_shift, n_step, cfg, seed):
	try:
		prompt_audio = _normalize_audio_input(prompt_audio)
		target_audio = _normalize_audio_input(target_audio)
		if not prompt_audio or not target_audio:
			gr.Warning(_i18n("warn_missing_audio"))
			return None

		session_base = _session_dir()
		params = SVCRunParams(
			prompt_vocal_sep=bool(prompt_vocal_sep),
			target_vocal_sep=bool(target_vocal_sep),
			auto_shift=bool(auto_shift),
			auto_mix_acc=bool(auto_mix_acc),
			pitch_shift=int(pitch_shift),
			n_steps=int(n_step),
			cfg=float(cfg),
			seed=int(seed),
			prompt_max_sec=PROMPT_MAX_SEC_DEFAULT,
			target_max_sec=TARGET_MAX_SEC_DEFAULT,
		)
		ok, msg, generated = APP_STATE.run_from_raw_audio_files(
			str(prompt_audio),
			str(target_audio),
			session_base,
			params,
		)
		if not ok or generated is None:
			print(msg, file=sys.stderr, flush=True)
			return None
		return str(generated)
	except Exception:
		_print_exception("_start_svc")
		return None


def render_interface() -> gr.Blocks:
	with gr.Blocks(title="SoulX-Singer-SVC Demo", theme=gr.themes.Default()) as page:
		gr.HTML(
            '<div style="'
            'text-align: center; '
            'padding: 1.25rem 0 1.5rem; '
            'margin-bottom: 0.5rem;'
            '">'
            '<div style="'
            'display: inline-block; '
            'font-size: 1.75rem; '
            'font-weight: 700; '
            'letter-spacing: 0.02em; '
            'color: #1a1a2e; '
            'line-height: 1.3;'
            '">SoulX-Singer-SVC</div>'
            '<div style="'
            'width: 80px; '
            'height: 3px; '
            'margin: 1rem auto 0; '
            'background: linear-gradient(90deg, transparent, #6366f1, transparent); '
            'border-radius: 2px;'
            '"></div>'
            '</div>'
        )
		with gr.Row(equal_height=True):
			lang_choice = gr.Radio(
				choices=["中文", "English"],
				value="中文",
				label=_i18n("display_lang_label"),
				type="index",
				interactive=True,
			)

		usage_md = gr.Markdown(_usage_md())

		with gr.Row(equal_height=True):
			prompt_audio = gr.Audio(
				label=_i18n("prompt_audio_label"),
				type="filepath",
				editable=False,
				interactive=True,
			)
			target_audio = gr.Audio(
				label=_i18n("target_audio_label"),
				type="filepath",
				editable=False,
				interactive=True,
			)

		with gr.Row(equal_height=True):
			prompt_vocal_sep = gr.Checkbox(label=_i18n("prompt_vocal_sep_label"), value=False, scale=1)
			target_vocal_sep = gr.Checkbox(label=_i18n("target_vocal_sep_label"), value=True, scale=1)
			auto_shift = gr.Checkbox(label=_i18n("auto_shift_label"), value=True, scale=1)
			auto_mix_acc = gr.Checkbox(label=_i18n("auto_mix_acc_label"), value=True, scale=1)

		with gr.Row(equal_height=True):
			pitch_shift = gr.Slider(label=_i18n("pitch_shift_label"), value=0, minimum=-36, maximum=36, step=1, scale=1)
			n_step = gr.Slider(label=_i18n("n_step_label"), value=32, minimum=1, maximum=200, step=1, scale=1)
			cfg = gr.Slider(label=_i18n("cfg_label"), value=1.0, minimum=0.0, maximum=10.0, step=0.1, scale=1)
			seed_input = gr.Slider(label=_i18n("seed_label"), value=42, minimum=0, maximum=10000, step=1, scale=1)

		with gr.Row():
			run_btn = gr.Button(value=_i18n("run_btn"), variant="primary", size="lg")

		with gr.Row():
			output_audio = gr.Audio(label=_i18n("output_audio_label"), type="filepath", interactive=False)

		gr.Examples(
			examples=EXAMPLE_LIST,
			inputs=[prompt_audio, target_audio],
			label=_i18n("examples_label"),
		)

		tips_md = gr.Markdown(_tips_md())

		run_btn.click(
			fn=_start_svc,
			inputs=[
				prompt_audio,
				target_audio,
				prompt_vocal_sep,
				target_vocal_sep,
				auto_shift,
				auto_mix_acc,
				pitch_shift,
				n_step,
				cfg,
				seed_input,
			],
			outputs=[output_audio],
		)

		def _change_language(lang):
			global _GLOBAL_LANG
			_GLOBAL_LANG = ["zh", "en"][lang]
			return [
				gr.update(label=_i18n("display_lang_label")),
				gr.update(value=_i18n("title")),
				gr.update(value=_usage_md()),
				gr.update(label=_i18n("prompt_audio_label")),
				gr.update(label=_i18n("target_audio_label")),
				gr.update(label=_i18n("prompt_vocal_sep_label")),
				gr.update(label=_i18n("target_vocal_sep_label")),
				gr.update(label=_i18n("auto_shift_label")),
				gr.update(label=_i18n("auto_mix_acc_label")),
				gr.update(label=_i18n("pitch_shift_label")),
				gr.update(label=_i18n("n_step_label")),
				gr.update(label=_i18n("cfg_label")),
				gr.update(label=_i18n("seed_label")),
				gr.update(value=_i18n("run_btn")),
				gr.update(label=_i18n("output_audio_label")),
				gr.update(value=_tips_md()),
			]

		lang_choice.change(
			fn=_change_language,
			inputs=[lang_choice],
			outputs=[
				lang_choice,
				usage_md,
				prompt_audio,
				target_audio,
				prompt_vocal_sep,
				target_vocal_sep,
				auto_shift,
				auto_mix_acc,
				pitch_shift,
				n_step,
				cfg,
				seed_input,
				run_btn,
				output_audio,
				tips_md,
			],
		)

	return page


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--port", type=int, default=7861, help="Gradio server port")
	parser.add_argument("--share", action="store_true", help="Create public link")
	parser.add_argument("--fp16", action="store_true", help="Use FP16 for SVC model and inference")
	args = parser.parse_args()

	page = render_interface()
	page.queue()
	page.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
