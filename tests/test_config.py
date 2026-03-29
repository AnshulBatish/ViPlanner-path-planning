"""Unit tests for configuration loading."""

from offroad_autonomy.utils.config import load_config


def test_load_config_reads_viplanner_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "beamng:",
                "  camera:",
                "    render_depth: true",
                "planning:",
                "  mode: viplanner",
                "viplanner:",
                '  model_dir: "models/viplanner"',
                "auto_goal:",
                "  fallback_forward_m: 3.5",
                "pure_pursuit:",
                "  wheelbase_m: 2.7",
                "perception:",
                '  model_weights: "dummy.pt"',
                "  prompts:",
                '    - "trail"',
                '    - "road"',
                "visualization:",
                "  dashboard:",
                "    colors:",
                "      BG: [1, 2, 3]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.beamng_render_depth is True
    assert config.planning_mode == "viplanner"
    assert config.viplanner_model_dir == "models/viplanner"
    assert config.auto_goal_fallback_forward_m == 3.5
    assert config.pure_pursuit_wheelbase_m == 2.7
    assert config.model_weights == "dummy.pt"
    assert config.perception_prompts == ["trail", "road"]
    assert config.dashboard_colors["BG"] == (1, 2, 3)
