"""
Helper script to render all Manim animations.

Usage:
    python render_all.py                    # Render all animations
    python render_all.py --phase 0          # Render Phase 0 only
    python render_all.py --phase 0.5        # Render Phase 0.5 (HIGHEST PRIORITY)
    python render_all.py --phase 1          # Render Phase 1 only
    python render_all.py --quality low     # Low quality (fast preview)
    python render_all.py --quality high    # High quality (final render)
"""

import subprocess
import sys
import os
import argparse

ANIMATIONS_DIR = os.path.join(os.path.dirname(__file__), "manim_animations")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

QUALITY_FLAGS = {
    "low": ["-ql"],
    "medium": ["-qm"],
    "high": ["-qh"],
    "production": ["-qp"],
}

SCENES = {
    0: {
        "file": "phase_0_rbf_derivation.py",
        "scenes": [
            "Distance_Concept",
            "Squared_Distance",
            "Gaussian_Function",
            "Gamma_Parameter",
            "RBF_Kernel_Construction",
            "Kernel_Trick",
        ],
        "output": "rbf_derivation",
    },
    0.5: {
        "file": "phase_0_5_kernel_mapping_3d.py",
        "scenes": [
            "Concentric_Circle_Dataset",
            "Failed_Linear_Separation",
            "Kernel_Lifting_To_3D",
            "Hyperplane_And_Margin",
            "Projection_Back_To_2D",
        ],
        "output": "kernel_mapping_3d",
    },
    1: {
        "file": "phase_1_svm_concepts.py",
        "scenes": [
            "Hyperplane_Theory",
            "Margin_Theory",
            "Support_Vector_Theory",
            "Soft_Margin_Theory",
            "Kernel_SVM_Overview",
            "RBF_SVM_Explanation",
        ],
        "output": "svm_concept_animation",
    },
}


def render_scene(filepath, scene_name, quality, output_dir):
    cmd = [
        "manim",
        filepath,
        scene_name,
        *QUALITY_FLAGS.get(quality, ["-ql"]),
        "--output_file", scene_name,
        "--media_dir", output_dir,
    ]
    print(f"\n{'='*60}")
    print(f"Rendering: {scene_name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Render Manim Animations")
    parser.add_argument(
        "--phase",
        type=float,
        choices=[0, 0.5, 1],
        help="Render specific phase (0, 0.5, or 1)",
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=list(QUALITY_FLAGS.keys()),
        default="medium",
        help="Render quality (default: medium)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Render a specific scene name",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases_to_render = (
        [args.phase] if args.phase is not None
        else sorted(SCENES.keys())
    )

    for phase_id in phases_to_render:
        if phase_id not in SCENES:
            print(f"[SKIP] Phase {phase_id} not found")
            continue

        phase = SCENES[phase_id]
        filepath = os.path.join(ANIMATIONS_DIR, phase["file"])

        if not os.path.exists(filepath):
            print(f"[SKIP] File not found: {filepath}")
            continue

        scenes_to_render = (
            [args.scene] if args.scene
            else phase["scenes"]
        )

        for scene_name in scenes_to_render:
            if scene_name not in phase["scenes"]:
                print(f"[SKIP] Scene '{scene_name}' not in Phase {phase_id}")
                continue
            success = render_scene(filepath, scene_name, args.quality, OUTPUT_DIR)
            status = "[OK]" if success else "[FAIL]"
            print(f"{status} {scene_name}")

    print(f"\n{'='*60}")
    print("All renders complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
