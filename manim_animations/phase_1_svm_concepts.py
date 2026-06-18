"""
Phase 1: SVM Concept Animation
===============================
Explains SVM theory after students understand geometric intuition.

Scenes:
  1. Hyperplane_Theory       - separating hyperplane concept
  2. Margin_Theory           - maximum margin principle
  3. Support_Vector_Theory   - critical samples
  4. Soft_Margin_Theory      - handling non-separable data
  5. Kernel_SVM_Overview     - putting it all together
  6. RBF_SVM_Explanation     - RBF kernel with SVM
"""

from manim import *
import numpy as np

np.random.seed(42)


class Hyperplane_Theory(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Separating Hyperplane", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=6,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x₁", y_label="x₂")
        self.play(Create(axes), Write(axes_labels))

        pos_points = np.array([[1, 1], [1.5, 2], [2, 1.5], [1.2, 2.5], [0.8, 1.8]])
        neg_points = np.array([[3.5, 3.5], [4, 3], [3, 4], [3.8, 2.5], [2.8, 3.2]])

        pos_dots = VGroup()
        for p in pos_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=BLUE, radius=0.1)
            pos_dots.add(dot)
        neg_dots = VGroup()
        for p in neg_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=RED, radius=0.1)
            neg_dots.add(dot)

        self.play(
            LaggedStart(*[Create(d) for d in pos_dots], lag_ratio=0.1),
            LaggedStart(*[Create(d) for d in neg_dots], lag_ratio=0.1),
        )
        self.wait(0.5)

        line = Line(
            axes.coords_to_point(0.5, 4.5),
            axes.coords_to_point(4.5, 0.5),
            color=GREEN, stroke_width=4,
        )
        line_label = MathTex(r"w^T x + b = 0", color=GREEN, font_size=32).next_to(line, UR, buff=0.2)
        self.play(Create(line), Write(line_label))
        self.wait(1)

        eq_def = MathTex(
            r"h(x) = \text{sign}(w^T x + b)",
            font_size=32, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(eq_def))
        self.wait(2)

        classification = VGroup(
            Text("wᵀx + b > 0  →  Class +1 (Blue)", font_size=26, color=BLUE),
            Text("wᵀx + b < 0  →  Class -1 (Red)", font_size=26, color=RED),
        ).arrange(DOWN).next_to(eq_def, DOWN, buff=0.3)

        self.play(Write(classification))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Margin_Theory(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Maximum Margin Principle", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=6,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x₁", y_label="x₂")
        self.play(Create(axes), Write(axes_labels))

        pos_points = np.array([[1, 1], [1.5, 1.8], [2.2, 1.2]])
        neg_points = np.array([[3.5, 3.5], [3, 4.2], [2.8, 3]])

        pos_dots = VGroup()
        for p in pos_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=BLUE, radius=0.1)
            pos_dots.add(dot)
        neg_dots = VGroup()
        for p in neg_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=RED, radius=0.1)
            neg_dots.add(dot)
        self.play(Create(pos_dots), Create(neg_dots))
        self.wait(0.3)

        hyperplane = Line(
            axes.coords_to_point(1, 4),
            axes.coords_to_point(4, 1),
            color=GREEN, stroke_width=4,
        )
        self.play(Create(hyperplane))
        self.wait(0.3)

        margin_pos = Line(
            axes.coords_to_point(0.5, 4.5),
            axes.coords_to_point(4.5, 0.5),
            color=BLUE, stroke_width=2, stroke_opacity=0.5,
        )
        margin_neg = Line(
            axes.coords_to_point(1.5, 3.5),
            axes.coords_to_point(3.5, 1.5),
            color=RED, stroke_width=2, stroke_opacity=0.5,
        )
        self.play(Create(margin_pos), Create(margin_neg))
        self.wait(0.3)

        brace = BraceBetweenPoints(
            axes.coords_to_point(2, 2.5),
            axes.coords_to_point(2.5, 2),
            color=YELLOW,
        )
        margin_label = Text("Margin", font_size=28, color=YELLOW).next_to(brace, RIGHT, buff=0.1)
        self.play(GrowFromCenter(brace), Write(margin_label))
        self.wait(0.5)

        many_lines = VGroup()
        for offset in np.linspace(-0.5, 0.5, 6):
            l = Line(
                axes.coords_to_point(1 + offset, 4 + offset),
                axes.coords_to_point(4 + offset, 1 + offset),
                color=GREY, stroke_width=1, stroke_opacity=0.3,
            )
            many_lines.add(l)
        self.play(Create(many_lines))
        self.wait(0.5)
        self.play(FadeOut(many_lines))

        best_text = Text(
            "SVM finds the line with maximum margin!",
            font_size=28, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(best_text))
        self.wait(2)

        objective = MathTex(
            r"\max_{w,b} \frac{2}{\|w\|} \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1",
            font_size=28, color=WHITE,
        ).next_to(best_text, DOWN, buff=0.3)
        self.play(Write(objective))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Support_Vector_Theory(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Support Vectors", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=6,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x₁", y_label="x₂")
        self.play(Create(axes), Write(axes_labels))

        pos_points = np.array([[1, 1], [2, 1.2], [1.5, 2.5], [0.8, 2], [1.2, 0.8]])
        neg_points = np.array([[4, 3.5], [3.5, 4], [3, 3.2], [4.2, 3], [2.8, 4.2]])

        pos_dots = VGroup()
        for p in pos_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=BLUE, radius=0.1)
            pos_dots.add(dot)
        neg_dots = VGroup()
        for p in neg_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=RED, radius=0.1)
            neg_dots.add(dot)
        self.play(Create(pos_dots), Create(neg_dots))
        self.wait(0.5)

        hyperplane = Line(
            axes.coords_to_point(1, 4),
            axes.coords_to_point(4, 1),
            color=GREEN, stroke_width=4,
        )
        self.play(Create(hyperplane))
        self.wait(0.3)

        margin_pos = Line(
            axes.coords_to_point(0.5, 4.5),
            axes.coords_to_point(4.5, 0.5),
            color=BLUE, stroke_width=2, stroke_opacity=0.5,
        )
        margin_neg = Line(
            axes.coords_to_point(1.5, 3.5),
            axes.coords_to_point(3.5, 1.5),
            color=RED, stroke_width=2, stroke_opacity=0.5,
        )
        self.play(Create(margin_pos), Create(margin_neg))
        self.wait(0.3)

        sv_pos = [np.array([2, 1.2]), np.array([1.5, 2.5])]
        sv_neg = [np.array([3.5, 4]), np.array([3, 3.2])]
        sv_dots = VGroup()
        for p in sv_pos + sv_neg:
            sv = Dot(axes.coords_to_point(p[0], p[1]), color=YELLOW, radius=0.15)
            glow = Dot(axes.coords_to_point(p[0], p[1]), color=YELLOW, radius=0.25, fill_opacity=0.2)
            sv_dots.add(glow, sv)

            margin_point = None
            if list(p) in [list(v) for v in sv_pos]:
                proj = np.array([p[0] - 0.1, p[1] + 0.1])
            else:
                proj = np.array([p[0] + 0.1, p[1] - 0.1])

            conn = DashedLine(
                axes.coords_to_point(p[0], p[1]),
                axes.coords_to_point(proj[0], proj[1]),
                color=YELLOW, stroke_width=2,
            )
            sv_dots.add(conn)

        self.play(
            LaggedStart(*[Create(d) for d in sv_dots], lag_ratio=0.1),
        )
        self.wait(0.5)

        sv_def = Text(
            "Support Vectors: Points on the margin boundary",
            font_size=28, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(sv_def))
        self.wait(1)

        sv_importance = Text(
            "Only support vectors determine the hyperplane!",
            font_size=26, color=GREEN,
        ).next_to(sv_def, DOWN, buff=0.3)
        self.play(Write(sv_importance))
        self.wait(2)

        change_text = VGroup(
            Text("Remove non-support vectors → no change", font_size=24, color=GREY),
            Text("Remove a support vector → hyperplane changes!", font_size=24, color=YELLOW),
        ).arrange(DOWN).next_to(sv_importance, DOWN, buff=0.3)
        self.play(Write(change_text))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Soft_Margin_Theory(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Soft Margin SVM", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=6,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x₁", y_label="x₂")
        self.play(Create(axes), Write(axes_labels))

        pos_points = np.array([[1, 1], [2, 1.5], [1.2, 2.2], [0.8, 1.5]])
        neg_points = np.array([[3.5, 3.5], [4, 3], [3, 4], [3.8, 2.5], [2.5, 3.8]])
        outlier = np.array([[3.2, 2.2]])

        pos_dots = VGroup()
        for p in pos_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=BLUE, radius=0.1)
            pos_dots.add(dot)
        neg_dots = VGroup()
        for p in neg_points:
            dot = Dot(axes.coords_to_point(p[0], p[1]), color=RED, radius=0.1)
            neg_dots.add(dot)

        self.play(Create(pos_dots), Create(neg_dots))
        self.wait(0.5)

        outlier_dot = Dot(axes.coords_to_point(outlier[0, 0], outlier[0, 1]), color=BLUE, radius=0.12)
        outlier_cross = Cross(stroke_color=YELLOW, stroke_width=3).move_to(outlier_dot)
        self.play(Create(outlier_dot), Create(outlier_cross))
        self.wait(0.5)

        hard_margin = Line(
            axes.coords_to_point(0.8, 4.5),
            axes.coords_to_point(3.8, 1.5),
            color=RED, stroke_width=3,
        )
        hard_label = Text("Hard margin (fails)", font_size=24, color=RED).next_to(hard_margin, UP, buff=0.1)
        self.play(Create(hard_margin), Write(hard_label))
        self.wait(1)
        self.play(FadeOut(hard_margin), FadeOut(hard_label))

        soft_margin = Line(
            axes.coords_to_point(1, 4.2),
            axes.coords_to_point(4.2, 1),
            color=GREEN, stroke_width=4,
        )
        soft_label = Text("Soft margin (allows errors)", font_size=24, color=GREEN).next_to(soft_margin, DOWN, buff=0.1)
        self.play(Create(soft_margin), Write(soft_label))
        self.wait(0.5)

        penalty = MathTex(
            r"\min \frac{1}{2}\|w\|^2 + C \sum \xi_i",
            font_size=32, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(penalty))
        self.wait(0.5)

        c_explain = VGroup(
            Text("C controls the trade-off:", font_size=26, color=WHITE),
            Text("Large C → fewer misclassifications (hard margin)", font_size=24, color=BLUE),
            Text("Small C → wider margin, more tolerance (soft margin)", font_size=24, color=RED),
        ).arrange(DOWN).next_to(penalty, DOWN, buff=0.3)
        self.play(Write(c_explain))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Kernel_SVM_Overview(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Kernel SVM Overview", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        flow = VGroup(
            Text("1. Original Data (Nonlinear)", font_size=28, color=GREY),
            MathTex(r"\downarrow", font_size=36, color=WHITE),
            Text("2. Kernel Function", font_size=28, color=BLUE),
            MathTex(r"K(x, x') = \exp(-\gamma\|x - x'\|^2)", font_size=28, color=YELLOW),
            MathTex(r"\downarrow", font_size=36, color=WHITE),
            Text("3. Feature Space (Linear in 3D+)", font_size=28, color=GREEN),
            MathTex(r"\downarrow", font_size=36, color=WHITE),
            Text("4. SVM Finds Hyperplane in Feature Space", font_size=28, color=BLUE),
            MathTex(r"\downarrow", font_size=36, color=WHITE),
            Text("5. Nonlinear Boundary in Original Space", font_size=28, color=GREEN),
        ).arrange(DOWN, buff=0.15)

        self.play(Write(flow))
        self.wait(3)

        key_insight = SurroundingRectangle(
            Text(
                "Kernel SVM = SVM in feature space,\nwithout explicitly computing the mapping",
                font_size=26, color=YELLOW,
            ),
            color=YELLOW, buff=0.2,
        ).to_edge(DOWN)
        self.play(Create(key_insight))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class RBF_SVM_Explanation(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("RBF SVM: Putting It All Together", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        parts = VGroup(
            Text("RBF Kernel", font_size=32, color=BLUE),
            MathTex(r"\exp(-\gamma\|x - x'\|^2)", font_size=28, color=BLUE),
            Text("Measures similarity between points", font_size=24, color=GREY),
            Text("", font_size=12),
            Text("+", font_size=32, color=YELLOW),
            Text("", font_size=12),
            Text("SVM", font_size=32, color=RED),
            MathTex(r"\max \text{Margin}, \min \|w\|^2", font_size=28, color=RED),
            Text("Finds optimal separating hyperplane", font_size=24, color=GREY),
            Text("", font_size=12),
            Text("=", font_size=32, color=GREEN),
            Text("", font_size=12),
            Text("RBF SVM", font_size=36, color=GREEN),
            Text("Nonlinear decision boundaries", font_size=28, color=GREEN),
            Text("Works on complex real-world data", font_size=28, color=GREEN),
        ).arrange(DOWN, buff=0.15)

        self.play(Write(parts))
        self.wait(2)

        gamma_guide = VGroup(
            Text("Gamma Guide:", font_size=28, color=YELLOW),
            Text("γ too small → underfitting (too smooth)", font_size=24, color=BLUE),
            Text("γ too large → overfitting (too wiggly)", font_size=24, color=RED),
            Text("γ = 1/n_features is a good default", font_size=24, color=GREEN),
        ).arrange(DOWN).to_edge(DOWN, buff=0.2)

        self.play(Write(gamma_guide))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )
