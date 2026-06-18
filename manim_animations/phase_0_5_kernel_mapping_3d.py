"""
Phase 0.5: Kernel Mapping 3D Demonstration
============================================
HIGHEST PRIORITY - Centerpiece of the entire project.

Demonstrates how the RBF kernel maps non-linearly separable 2D data
(concentric circles) into 3D where a linear hyperplane can separate classes.

Scene flow:
  1. Concentric_Circle_Dataset  - 2D non-separable data
  2. Failed_Linear_Separation   - linear classifiers fail
  3. Introduce_Feature_Mapping  - concept of higher dimensions
  4. Kernel_Lifting_To_3D       - z = x² + y² transformation
  5. Camera_Rotation            - 3D spatial understanding
  6. Hyperplane_Creation        - separating surface
  7. Support_Vector_Visualization - critical samples
  8. Margin_Visualization       - maximum margin
  9. Projection_Back_To_2D     - nonlinear decision boundary
"""

from manim import *
import numpy as np

np.random.seed(42)

CONFIG = {
    "inner_radius": 0.35,
    "outer_radius": 0.85,
    "inner_noise": 0.08,
    "outer_noise": 0.08,
    "n_inner": 60,
    "n_outer": 60,
    "inner_color": BLUE,
    "outer_color": RED,
    "plane_center": 0.35,
    "camera_phi": 70 * DEGREES,
    "camera_theta": -45 * DEGREES,
    "camera_gamma": 0 * DEGREES,
    "camera_zoom": 0.7,
    "rotation_rate": 0.15,
    "point_radius": 0.06,
    "sv_radius": 0.09,
}


def generate_concentric_data(config=None):
    if config is None:
        config = CONFIG
    n_inner = config["n_inner"]
    n_outer = config["n_outer"]

    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    r_inner = config["inner_radius"] + np.random.uniform(
        -config["inner_noise"], config["inner_noise"], n_inner
    )
    inner_x = r_inner * np.cos(theta_inner)
    inner_y = r_inner * np.sin(theta_inner)

    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    r_outer = config["outer_radius"] + np.random.uniform(
        -config["outer_noise"], config["outer_noise"], n_outer
    )
    outer_x = r_outer * np.cos(theta_outer)
    outer_y = r_outer * np.sin(theta_outer)

    inner = np.column_stack([inner_x, inner_y])
    outer = np.column_stack([outer_x, outer_y])
    return inner, outer


def compute_z(points_2d):
    return points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2


def get_support_vectors(inner_3d, outer_3d, plane_z=0.35):
    all_inner = np.column_stack([inner_3d, np.full(len(inner_3d), plane_z)])
    all_outer = np.column_stack([outer_3d, np.full(len(outer_3d), plane_z)])
    inner_dists = np.abs(inner_3d[:, 2] - plane_z)
    outer_dists = np.abs(outer_3d[:, 2] - plane_z)
    n_sv = min(5, len(inner_dists))
    inner_sv_idx = np.argsort(inner_dists)[:n_sv]
    outer_sv_idx = np.argsort(outer_dists)[:n_sv]
    return inner_sv_idx, outer_sv_idx


class Concentric_Circle_Dataset(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"
        inner, outer = generate_concentric_data()

        axes_2d = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=8,
            y_length=8,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_2d_labels = axes_2d.get_axis_labels(x_label="x", y_label="y")
        title = Text("Concentric Circle Dataset", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.play(Create(axes_2d), Write(axes_2d_labels))
        self.wait(0.3)

        inner_dots = VGroup()
        for p in inner:
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["inner_color"])
            inner_dots.add(dot)
        outer_dots = VGroup()
        for p in outer:
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["outer_color"])
            outer_dots.add(dot)

        inner_label = Text("Class A", font_size=28, color=CONFIG["inner_color"]).to_corner(UL)
        outer_label = Text("Class B", font_size=28, color=CONFIG["outer_color"]).to_corner(UR)

        self.play(
            LaggedStart(
                *[Create(d, scale=0.5) for d in inner_dots],
                lag_ratio=0.02,
            ),
            LaggedStart(
                *[Create(d, scale=0.5) for d in outer_dots],
                lag_ratio=0.02,
            ),
            Write(inner_label),
            Write(outer_label),
        )
        self.wait(1)

        question = Text(
            "Can we separate these with a straight line?",
            font_size=28, color=YELLOW,
        ).next_to(title, DOWN, buff=0.3)
        self.play(Write(question))
        self.wait(1.5)
        self.play(FadeOut(question))
        self.wait(0.5)

        self.inner_dots_2d = inner_dots
        self.outer_dots_2d = outer_dots
        self.inner_pts = inner
        self.outer_pts = outer
        self.axes_2d = axes_2d
        self.axes_2d_labels = axes_2d_labels
        self.title = title
        self.inner_label = inner_label
        self.outer_label = outer_label


class Failed_Linear_Separation(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"
        inner, outer = generate_concentric_data()

        axes_2d = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=8,
            y_length=8,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_2d_labels = axes_2d.get_axis_labels(x_label="x", y_label="y")
        title = Text("Linear Separation Attempts", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title), Create(axes_2d), Write(axes_2d_labels))

        inner_dots = VGroup()
        for p in inner:
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["inner_color"])
            inner_dots.add(dot)
        outer_dots = VGroup()
        for p in outer:
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["outer_color"])
            outer_dots.add(dot)
        inner_label = Text("Class A", font_size=28, color=CONFIG["inner_color"]).to_corner(UL)
        outer_label = Text("Class B", font_size=28, color=CONFIG["outer_color"]).to_corner(UR)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in inner_dots], lag_ratio=0.02),
            LaggedStart(*[Create(d, scale=0.5) for d in outer_dots], lag_ratio=0.02),
            Write(inner_label), Write(outer_label),
        )
        self.wait(0.5)

        attempts = [
            {"angle": 0, "label": "Horizontal Line"},
            {"angle": 45 * DEGREES, "label": "Diagonal Line"},
            {"angle": 90 * DEGREES, "label": "Vertical Line"},
            {"angle": 30 * DEGREES, "label": "Angled Line"},
            {"angle": 60 * DEGREES, "label": "Steep Line"},
        ]

        fail_texts = VGroup()
        for attempt in attempts:
            angle = attempt["angle"]
            line = DashedLine(
                start=axes_2d.coords_to_point(-1.5 * np.cos(angle), -1.5 * np.sin(angle)),
                end=axes_2d.coords_to_point(1.5 * np.cos(angle), 1.5 * np.sin(angle)),
                color=WHITE, stroke_width=3,
            )
            label = Text(attempt["label"], font_size=22, color=WHITE).next_to(line, UP, buff=0.1)
            self.play(Create(line), Write(label), run_time=0.5)

            fail_dots = VGroup()
            for p in inner:
                pt = axes_2d.coords_to_point(p[0], p[1])
                side = p[0] * np.sin(angle) - p[1] * np.cos(angle)
                if side > 0:
                    fd = Dot(pt, radius=CONFIG["point_radius"], color=WHITE)
                    fail_dots.add(fd)
            for p in outer:
                pt = axes_2d.coords_to_point(p[0], p[1])
                side = p[0] * np.sin(angle) - p[1] * np.cos(angle)
                if side < 0:
                    fd = Dot(pt, radius=CONFIG["point_radius"], color=WHITE)
                    fail_dots.add(fd)

            if len(fail_dots) > 0:
                self.play(
                    LaggedStart(*[d.animate.set_color(YELLOW) for d in fail_dots], lag_ratio=0.02),
                )
                self.wait(0.3)
                self.play(
                    LaggedStart(*[d.animate.set_color(WHITE) for d in fail_dots], lag_ratio=0.02),
                )

            self.play(FadeOut(line), FadeOut(label), run_time=0.3)

        conclusion = Text(
            "Linear classifiers cannot separate concentric data!",
            font_size=28, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(1.5)
        self.play(FadeOut(conclusion), FadeOut(title))
        self.wait(0.3)

        self.inner_dots_2d = inner_dots
        self.outer_dots_2d = outer_dots
        self.inner_pts = inner
        self.outer_pts = outer
        self.axes_2d = axes_2d


class Kernel_Lifting_To_3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"
        self.set_camera_orientation(
            phi=CONFIG["camera_phi"],
            theta=CONFIG["camera_theta"],
            gamma=CONFIG["camera_gamma"],
            zoom=CONFIG["camera_zoom"],
        )

        inner, outer = generate_concentric_data()
        z_inner = compute_z(inner)
        z_outer = compute_z(outer)

        inner_min_z, inner_max_z = z_inner.min(), z_inner.max()
        outer_min_z, outer_max_z = z_outer.min(), z_outer.max()
        all_z = np.concatenate([z_inner, z_outer])
        z_min, z_max = all_z.min(), all_z.max()
        z_range = z_max - z_min

        axes_2d = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=6,
            y_length=6,
            axis_config={"color": GREY},
        )
        axes_2d_labels = axes_2d.get_axis_labels(x_label="x", y_label="y")
        axes_2d.move_to(ORIGIN)

        title = Text("Kernel Mapping: z = x² + y²", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.play(Create(axes_2d), Write(axes_2d_labels))

        inner_dots_2d = VGroup()
        for i, p in enumerate(inner):
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["inner_color"])
            inner_dots_2d.add(dot)
        outer_dots_2d = VGroup()
        for i, p in enumerate(outer):
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["outer_color"])
            outer_dots_2d.add(dot)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in inner_dots_2d], lag_ratio=0.01),
            LaggedStart(*[Create(d, scale=0.5) for d in outer_dots_2d], lag_ratio=0.01),
        )
        self.wait(0.5)

        mapping_eq = MathTex(
            r"z = \phi(x, y) = x^2 + y^2",
            font_size=40, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(mapping_eq))
        self.wait(1)

        self.play(
            FadeOut(axes_2d),
            FadeOut(axes_2d_labels),
            *[FadeOut(d) for d in inner_dots_2d],
            *[FadeOut(d) for d in outer_dots_2d],
            mapping_eq.animate.scale(0.8).to_corner(UR),
        )

        axes_3d = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[0, 1.5, 0.25],
            x_length=7,
            y_length=7,
            z_length=5,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_3d_labels = axes_3d.get_axis_labels(x_label="x", y_label="y", z_label="z")
        self.play(
            Create(axes_3d),
            Write(axes_3d_labels),
        )
        self.wait(0.5)

        inner_dots_3d = VGroup()
        for i, p in enumerate(inner):
            z_val = z_inner[i]
            dot = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], 0),
                radius=CONFIG["point_radius"],
                color=CONFIG["inner_color"],
            )
            inner_dots_3d.add(dot)
        outer_dots_3d = VGroup()
        for i, p in enumerate(outer):
            z_val = z_outer[i]
            dot = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], 0),
                radius=CONFIG["point_radius"],
                color=CONFIG["outer_color"],
            )
            outer_dots_3d.add(dot)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in inner_dots_3d], lag_ratio=0.01),
            LaggedStart(*[Create(d, scale=0.5) for d in outer_dots_3d], lag_ratio=0.01),
        )
        self.wait(0.5)

        self.begin_ambient_camera_rotation(rate=CONFIG["rotation_rate"])

        lift_anims_inner = []
        for i, dot in enumerate(inner_dots_3d):
            z_val = z_inner[i]
            target = axes_3d.coords_to_point(inner[i, 0], inner[i, 1], z_val)
            lift_anims_inner.append(dot.animate.move_to(target))
        lift_anims_outer = []
        for i, dot in enumerate(outer_dots_3d):
            z_val = z_outer[i]
            target = axes_3d.coords_to_point(outer[i, 0], outer[i, 1], z_val)
            lift_anims_outer.append(dot.animate.move_to(target))

        lift_text = Text(
            "Mapping to 3D Feature Space...",
            font_size=28, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(lift_text), run_time=0.3)

        self.play(
            LaggedStart(*lift_anims_inner, lag_ratio=0.015),
            LaggedStart(*lift_anims_outer, lag_ratio=0.015),
            run_time=3,
        )
        self.wait(0.5)
        self.play(FadeOut(lift_text))
        self.wait(1)

        separable_text = Text(
            "Now linearly separable in 3D!",
            font_size=32, color=GREEN,
        ).to_edge(DOWN)
        self.play(Write(separable_text))
        self.wait(2)
        self.play(FadeOut(separable_text))

        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        self.inner_dots_3d = inner_dots_3d
        self.outer_dots_3d = outer_dots_3d
        self.inner_pts = inner
        self.outer_pts = outer
        self.z_inner = z_inner
        self.z_outer = z_outer
        self.axes_3d = axes_3d
        self.axes_3d_labels = axes_3d_labels
        self.mapping_eq = mapping_eq
        self.title = title


class Hyperplane_And_Margin(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"
        self.set_camera_orientation(
            phi=CONFIG["camera_phi"],
            theta=CONFIG["camera_theta"],
            gamma=CONFIG["camera_gamma"],
            zoom=CONFIG["camera_zoom"],
        )

        inner, outer = generate_concentric_data()
        z_inner = compute_z(inner)
        z_outer = compute_z(outer)

        axes_3d = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[0, 1.5, 0.25],
            x_length=7,
            y_length=7,
            z_length=5,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_3d_labels = axes_3d.get_axis_labels(x_label="x", y_label="y", z_label="z")

        title = Text("Hyperplane & Support Vectors", font_size=36, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.play(Create(axes_3d), Write(axes_3d_labels))
        self.wait(0.3)

        inner_dots_3d = VGroup()
        for i, p in enumerate(inner):
            z_val = z_inner[i]
            dot = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_val),
                radius=CONFIG["point_radius"],
                color=CONFIG["inner_color"],
            )
            inner_dots_3d.add(dot)
        outer_dots_3d = VGroup()
        for i, p in enumerate(outer):
            z_val = z_outer[i]
            dot = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_val),
                radius=CONFIG["point_radius"],
                color=CONFIG["outer_color"],
            )
            outer_dots_3d.add(dot)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in inner_dots_3d], lag_ratio=0.01),
            LaggedStart(*[Create(d, scale=0.5) for d in outer_dots_3d], lag_ratio=0.01),
        )
        self.wait(0.5)

        self.begin_ambient_camera_rotation(rate=CONFIG["rotation_rate"])

        self.wait(1)

        plane_z = CONFIG["plane_center"]
        plane = Surface(
            lambda u, v: axes_3d.coords_to_point(u, v, plane_z),
            u_range=[-1.2, 1.2],
            v_range=[-1.2, 1.2],
            resolution=(20, 20),
            fill_color=GREEN,
            fill_opacity=0.3,
            stroke_color=GREEN,
            stroke_width=0.5,
            stroke_opacity=0.2,
        )

        plane_label = MathTex(
            r"ax + by + cz + d = 0",
            font_size=32, color=GREEN,
        ).to_corner(DL)
        plane_label_shifted = MathTex(
            r"z = \text{constant}",
            font_size=28, color=GREEN,
        ).next_to(plane_label, DOWN, buff=0.2)

        self.play(
            Create(plane, run_time=2),
            Write(plane_label),
            Write(plane_label_shifted),
        )
        self.wait(1.5)

        inner_sv_idx, outer_sv_idx = get_support_vectors(
            inner, outer, plane_z
        )

        sv_text = Text("Support Vectors", font_size=28, color=YELLOW).to_corner(DR)
        self.play(Write(sv_text))

        sv_group = VGroup()
        for idx in inner_sv_idx:
            p = inner[idx]
            z_val = z_inner[idx]
            sv = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_val),
                radius=CONFIG["sv_radius"],
                color=YELLOW,
            )
            glow = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_val),
                radius=CONFIG["sv_radius"] * 2,
                color=YELLOW,
                fill_opacity=0.2,
            )
            sv_group.add(glow, sv)

            line_start = axes_3d.coords_to_point(p[0], p[1], z_val)
            line_end = axes_3d.coords_to_point(p[0], p[1], plane_z)
            connection = DashedLine(line_start, line_end, color=YELLOW, stroke_width=2)
            sv_group.add(connection)

        for idx in outer_sv_idx:
            p = outer[idx]
            z_val = z_outer[idx]
            sv = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_val),
                radius=CONFIG["sv_radius"],
                color=YELLOW,
            )
            glow = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_val),
                radius=CONFIG["sv_radius"] * 2,
                color=YELLOW,
                fill_opacity=0.2,
            )
            sv_group.add(glow, sv)

            line_start = axes_3d.coords_to_point(p[0], p[1], z_val)
            line_end = axes_3d.coords_to_point(p[0], p[1], plane_z)
            connection = DashedLine(line_start, line_end, color=YELLOW, stroke_width=2)
            sv_group.add(connection)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in sv_group], lag_ratio=0.05),
            run_time=2,
        )
        self.wait(2)

        margin_text = Text(
            "Maximum Margin",
            font_size=28, color=ORANGE,
        ).next_to(sv_text, DOWN, buff=0.3)
        self.play(Write(margin_text))

        upper_plane = Surface(
            lambda u, v: axes_3d.coords_to_point(u, v, plane_z + 0.15),
            u_range=[-1.2, 1.2],
            v_range=[-1.2, 1.2],
            resolution=(10, 10),
            fill_color=ORANGE,
            fill_opacity=0.15,
            stroke_color=ORANGE,
            stroke_width=0.3,
            stroke_opacity=0.1,
        )
        lower_plane = Surface(
            lambda u, v: axes_3d.coords_to_point(u, v, plane_z - 0.15),
            u_range=[-1.2, 1.2],
            v_range=[-1.2, 1.2],
            resolution=(10, 10),
            fill_color=ORANGE,
            fill_opacity=0.15,
            stroke_color=ORANGE,
            stroke_width=0.3,
            stroke_opacity=0.1,
        )

        self.play(
            Create(upper_plane, run_time=1),
            Create(lower_plane, run_time=1),
        )
        self.wait(3)

        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        self.play(
            FadeOut(sv_text),
            FadeOut(margin_text),
            FadeOut(plane_label),
            FadeOut(plane_label_shifted),
            FadeOut(plane),
            FadeOut(upper_plane),
            FadeOut(lower_plane),
            *[FadeOut(d) for d in sv_group],
        )
        self.wait(0.3)

        self.inner_dots_3d = inner_dots_3d
        self.outer_dots_3d = outer_dots_3d
        self.inner_pts = inner
        self.outer_pts = outer
        self.axes_3d = axes_3d


class Projection_Back_To_2D(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"
        self.set_camera_orientation(
            phi=CONFIG["camera_phi"],
            theta=CONFIG["camera_theta"],
            gamma=CONFIG["camera_gamma"],
            zoom=CONFIG["camera_zoom"],
        )

        inner, outer = generate_concentric_data()
        z_inner = compute_z(inner)
        z_outer = compute_z(outer)

        axes_3d = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[0, 1.5, 0.25],
            x_length=7,
            y_length=7,
            z_length=5,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_3d_labels = axes_3d.get_axis_labels(x_label="x", y_label="y", z_label="z")

        title = Text(
            "Projection: Nonlinear Decision Boundary",
            font_size=36, color=WHITE,
        ).to_edge(UP)
        self.play(Write(title))
        self.play(Create(axes_3d), Write(axes_3d_labels))

        inner_dots_3d = VGroup()
        for i, p in enumerate(inner):
            dot = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_inner[i]),
                radius=CONFIG["point_radius"],
                color=CONFIG["inner_color"],
            )
            inner_dots_3d.add(dot)
        outer_dots_3d = VGroup()
        for i, p in enumerate(outer):
            dot = Dot3D(
                axes_3d.coords_to_point(p[0], p[1], z_outer[i]),
                radius=CONFIG["point_radius"],
                color=CONFIG["outer_color"],
            )
            outer_dots_3d.add(dot)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in inner_dots_3d], lag_ratio=0.01),
            LaggedStart(*[Create(d, scale=0.5) for d in outer_dots_3d], lag_ratio=0.01),
        )
        self.wait(0.5)

        self.begin_ambient_camera_rotation(rate=CONFIG["rotation_rate"])

        self.wait(1)

        plane_z = CONFIG["plane_center"]
        plane = Surface(
            lambda u, v: axes_3d.coords_to_point(u, v, plane_z),
            u_range=[-1.2, 1.2],
            v_range=[-1.2, 1.2],
            resolution=(20, 20),
            fill_color=GREEN,
            fill_opacity=0.3,
            stroke_color=GREEN,
            stroke_width=0.5,
            stroke_opacity=0.2,
        )
        self.play(Create(plane, run_time=1.5))
        self.wait(1)

        project_text = Text(
            "Projecting hyperplane back to 2D...",
            font_size=28, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(project_text))
        self.wait(1)

        self.stop_ambient_camera_rotation()
        self.move_camera(
            phi=90 * DEGREES,
            theta=-90 * DEGREES,
            zoom=1.0,
            run_time=2,
        )
        self.wait(0.5)

        self.play(FadeOut(axes_3d_labels), FadeOut(plane))

        axes_2d = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=7,
            y_length=7,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_2d_labels = axes_2d.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes_2d), Write(axes_2d_labels))

        inner_dots_2d = VGroup()
        for i, p in enumerate(inner):
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["inner_color"])
            inner_dots_2d.add(dot)
        outer_dots_2d = VGroup()
        for i, p in enumerate(outer):
            dot = Dot(axes_2d.coords_to_point(p[0], p[1]), radius=CONFIG["point_radius"], color=CONFIG["outer_color"])
            outer_dots_2d.add(dot)

        self.play(
            LaggedStart(*[Create(d, scale=0.5) for d in inner_dots_2d], lag_ratio=0.02),
            LaggedStart(*[Create(d, scale=0.5) for d in outer_dots_2d], lag_ratio=0.02),
        )
        self.wait(0.5)

        circle_radius = np.sqrt(plane_z)
        boundary_circle = Circle(
            radius=circle_radius * 4,
            color=GREEN,
            stroke_width=4,
        )
        boundary_circle.move_to(axes_2d.coords_to_point(0, 0))
        self.play(Create(boundary_circle), run_time=1.5)

        boundary_label = Text(
            "Nonlinear Decision Boundary",
            font_size=28, color=GREEN,
        ).to_edge(DOWN)
        self.play(Write(boundary_label))
        self.wait(2)

        kernel_trick_text = MathTex(
            r"K(x, x') = \exp(-\gamma\|x - x'\|^2)",
            font_size=36, color=YELLOW,
        ).to_corner(UR)
        self.play(Write(kernel_trick_text))
        self.wait(1.5)

        self.play(FadeOut(project_text))
        final_message = Text(
            "RBF Kernel: Linearly inseparable → Separable!",
            font_size=32, color=GREEN,
        ).to_edge(DOWN, buff=0.5)
        self.play(
            FadeOut(boundary_label),
            Write(final_message),
        )
        self.wait(3)
