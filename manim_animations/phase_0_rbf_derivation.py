"""
Phase 0: Mathematical Foundation of RBF Kernel
===============================================
Builds mathematical intuition for similarity measurement and kernel functions.

Scenes:
  1. Distance_Concept        - Euclidean distance between two points
  2. Squared_Distance        - squared Euclidean distance
  3. Gaussian_Function       - exponential decay
  4. Gamma_Parameter         - effect of gamma on similarity
  5. RBF_Kernel_Construction - K(x,x') = exp(-gamma||x-x'||^2)
  6. Kernel_Trick            - implicit mapping via kernel
"""

from manim import *
import numpy as np

np.random.seed(42)


class Distance_Concept(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Euclidean Distance", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        self.play(Create(axes), Write(axes_labels))

        p1 = Dot(axes.coords_to_point(1, 1), color=BLUE, radius=0.1)
        p2 = Dot(axes.coords_to_point(4, 3), color=RED, radius=0.1)
        p1_label = MathTex(r"x", color=BLUE, font_size=32).next_to(p1, DL)
        p2_label = MathTex(r"x'", color=RED, font_size=32).next_to(p2, UR)

        self.play(Create(p1), Create(p2), Write(p1_label), Write(p2_label))
        self.wait(0.5)

        dist_line = DashedLine(
            p1.get_center(), p2.get_center(),
            color=YELLOW, stroke_width=3,
        )
        self.play(Create(dist_line))
        self.wait(0.3)

        dx = ValueTracker(3)
        dy = ValueTracker(2)
        dist_brace = BraceBetweenPoints(p1.get_center(), p2.get_center(), color=YELLOW)
        dist_brace_label = MathTex(
            r"d(x, x') = \|x - x'\|",
            font_size=36, color=YELLOW,
        ).next_to(dist_brace, DOWN, buff=0.3)

        self.play(
            GrowFromCenter(dist_brace),
            Write(dist_brace_label),
        )
        self.wait(2)

        formula = MathTex(
            r"d(x, x') = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2}",
            font_size=32, color=WHITE,
        ).to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)

        self.play(
            FadeOut(dist_line),
            FadeOut(dist_brace),
            FadeOut(dist_brace_label),
            FadeOut(p1_label),
            FadeOut(p2_label),
            FadeOut(p1),
            FadeOut(p2),
            FadeOut(axes),
            FadeOut(axes_labels),
            FadeOut(formula),
            FadeOut(title),
        )


class Squared_Distance(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Squared Euclidean Distance", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        eq1 = MathTex(
            r"\|x - x'\|^2 = (x_1 - x_1')^2 + (x_2 - x_2')^2 + \dots",
            font_size=36, color=WHITE,
        )
        self.play(Write(eq1))
        self.wait(1)

        note = Text(
            "Squares the distance, amplifying larger differences",
            font_size=28, color=YELLOW,
        ).next_to(eq1, DOWN, buff=0.5)
        self.play(Write(note))
        self.wait(2)

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=7,
            y_length=5,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        axes.shift(DOWN * 0.5)
        self.play(Create(axes), Write(axes_labels))
        self.wait(0.3)

        points = [
            (1, 1, BLUE, "Close"),
            (4, 3, RED, "Far"),
        ]
        dots = []
        for x, y, color, label in points:
            dot = Dot(axes.coords_to_point(x, y), color=color, radius=0.1)
            dots.append(dot)
        self.play(*[Create(d) for d in dots])
        self.wait(0.5)

        comparison = MathTex(
            r"\text{Small distance} \rightarrow \text{Small squared}",
            font_size=28, color=BLUE,
        ).to_corner(DL)
        comparison2 = MathTex(
            r"\text{Large distance} \rightarrow \text{Large squared}",
            font_size=28, color=RED,
        ).to_corner(DR)
        self.play(Write(comparison), Write(comparison2))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Gaussian_Function(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Gaussian / Exponential Function", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 1.2, 0.2],
            x_length=8,
            y_length=5,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label=r"f(x)")
        self.play(Create(axes), Write(axes_labels))

        graph_exp = axes.plot(
            lambda x: np.exp(-x),
            x_range=[0, 5],
            color=BLUE,
            stroke_width=4,
        )
        graph_label = MathTex(r"f(x) = e^{-x}", color=BLUE, font_size=36).next_to(graph_exp, UR, buff=0.2)
        self.play(Create(graph_exp), Write(graph_label))
        self.wait(0.5)

        x_tracker = ValueTracker(0.5)
        moving_dot = Dot(color=YELLOW)
        moving_dot.add_updater(
            lambda d: d.move_to(axes.coords_to_point(x_tracker.get_value(), np.exp(-x_tracker.get_value())))
        )
        self.add(moving_dot)

        value_text = always_redraw(
            lambda: MathTex(
                rf"e^{{-{x_tracker.get_value():.2f}}} = {np.exp(-x_tracker.get_value()):.3f}",
                font_size=28, color=YELLOW,
            ).to_edge(DOWN)
        )
        self.add(value_text)

        self.play(x_tracker.animate.set_value(3), run_time=3)
        self.wait(0.5)
        self.play(x_tracker.animate.set_value(0.5), run_time=2)
        self.wait(0.5)

        self.remove(moving_dot, value_text)

        note1 = Text("When distance is small → value near 1", font_size=28, color=GREEN).to_edge(DOWN)
        note2 = Text("When distance is large → value near 0", font_size=28, color=RED).next_to(note1, DOWN, buff=0.2)

        self.play(
            FadeOut(graph_label),
            Write(note1),
        )
        self.wait(1)
        self.play(Write(note2))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Gamma_Parameter(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Gamma Parameter: Influence Control", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 1.2, 0.2],
            x_length=8,
            y_length=5,
            axis_config={"color": GREY, "include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="Distance", y_label="Similarity")
        self.play(Create(axes), Write(axes_labels))

        gamma_values = [0.5, 1.0, 2.0, 5.0]
        gamma_colors = [BLUE, GREEN, YELLOW, RED]
        gamma_labels_text = []

        last_graph = None
        for i, (g, c) in enumerate(zip(gamma_values, gamma_colors)):
            graph = axes.plot(
                lambda x, gv=g: np.exp(-gv * x),
                x_range=[0, 5],
                color=c,
                stroke_width=3,
            )
            label = MathTex(
                rf"\gamma = {g}", color=c, font_size=28,
            ).next_to(axes, UR if i == 0 else DOWN, buff=0.1 * (i + 1))
            gamma_labels_text.append(label)

            if last_graph is None:
                self.play(
                    Create(graph),
                    Write(label),
                    run_time=0.8,
                )
            else:
                self.play(
                    Create(graph),
                    Write(label),
                    run_time=0.5,
                )
            self.wait(0.5)
            last_graph = graph

        self.wait(1)

        explanation = VGroup(
            Text("Small γ: Wide influence, smooth boundary", font_size=26, color=BLUE),
            Text("Large γ: Narrow influence, complex boundary", font_size=26, color=RED),
        ).arrange(DOWN, aligned_center=True).to_edge(DOWN)

        self.play(Write(explanation))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class RBF_Kernel_Construction(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("RBF Kernel Construction", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        concepts = VGroup(
            MathTex(r"1.", r"\text{ Distance: }", r"\|x - x'\|", font_size=32),
            MathTex(r"2.", r"\text{ Squared: }", r"\|x - x'\|^2", font_size=32),
            MathTex(r"3.", r"\text{ Scaled: }", r"\gamma \|x - x'\|^2", font_size=32),
            MathTex(r"4.", r"\text{ Exponentiated: }", r"e^{-\gamma \|x - x'\|^2}", font_size=32),
        ).arrange(DOWN, aligned_center=True).scale(0.9)

        self.play(Write(concepts))
        self.wait(2)

        arrow = MathTex(r"\Downarrow", font_size=40, color=YELLOW)
        arrow.next_to(concepts, DOWN, buff=0.3)
        self.play(Write(arrow))
        self.wait(0.5)

        kernel_eq = MathTex(
            r"K(x, x') = \exp(-\gamma \|x - x'\|^2)",
            font_size=44, color=YELLOW,
        )
        kernel_eq.next_to(arrow, DOWN, buff=0.3)
        self.play(Write(kernel_eq))
        self.wait(1)

        box = SurroundingRectangle(kernel_eq, color=GREEN, buff=0.2)
        self.play(Create(box))
        self.wait(1)

        properties = VGroup(
            Text("Symmetric: K(x,x') = K(x',x)", font_size=26, color=BLUE),
            Text("Positive semi-definite", font_size=26, color=BLUE),
            Text("Range: (0, 1]", font_size=26, color=BLUE),
            Text("Universal approximator", font_size=26, color=BLUE),
        ).arrange(DOWN, aligned_center=True).next_to(box, DOWN, buff=0.5)

        self.play(Write(properties))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
        )


class Kernel_Trick(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("The Kernel Trick", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(title))

        concept = Text(
            "What if we could compute dot products\nin a high-dimensional space...\nwithout ever going there?",
            font_size=28, color=WHITE,
        )
        self.play(Write(concept))
        self.wait(2)
        self.play(FadeOut(concept))

        explicit = MathTex(
            r"\text{Explicit: } \phi(x)^T \phi(x')",
            font_size=36, color=RED,
        )
        explicit.to_edge(LEFT).shift(UP)
        self.play(Write(explicit))
        self.wait(0.5)

        implicit = MathTex(
            r"\text{Implicit: } K(x, x')",
            font_size=36, color=GREEN,
        )
        implicit.to_edge(RIGHT).shift(UP)
        self.play(Write(implicit))
        self.wait(0.5)

        mapping_example = MathTex(
            r"\phi(x) = \begin{bmatrix} x_1^2 & x_2^2 & \sqrt{2}x_1x_2 \end{bmatrix}^T",
            font_size=28, color=BLUE,
        ).next_to(explicit, DOWN, buff=0.5)
        self.play(Write(mapping_example))
        self.wait(0.5)

        computation = MathTex(
            r"\phi(x)^T\phi(x') = (x^T x')^2 = K(x, x')",
            font_size=28, color=GREEN,
        ).next_to(implicit, DOWN, buff=0.5)
        self.play(Write(computation))
        self.wait(1)

        benefit_box = Rectangle(
            width=10, height=1.8,
            color=YELLOW, fill_opacity=0.1,
        ).to_edge(DOWN, buff=0.3)
        benefit_text = VGroup(
            Text("Benefits of Kernel Trick:", font_size=26, color=YELLOW),
            Text("• Compute high-dimensional dot products in O(n) instead of O(n²)",
                 font_size=22, color=WHITE),
            Text("• RBF kernel corresponds to infinite-dimensional mapping",
                 font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_center=True, buff=0.1).move_to(benefit_box)

        self.play(
            Create(benefit_box),
            Write(benefit_text),
            run_time=1.5,
        )
        self.wait(2)

        summary = VGroup(
            Text("Kernel Trick: Inner product in feature space",
                 font_size=30, color=YELLOW),
            Text("computed efficiently in input space.",
                 font_size=30, color=YELLOW),
        ).arrange(DOWN).move_to(ORIGIN)

        self.play(
            FadeOut(explicit),
            FadeOut(implicit),
            FadeOut(mapping_example),
            FadeOut(computation),
            FadeOut(benefit_box),
            FadeOut(benefit_text),
            Write(summary),
        )
        self.wait(2)
