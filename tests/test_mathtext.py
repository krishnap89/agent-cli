import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_cli import agent, coder, writing
from agent_cli.mathtext import plain_math


class TestPlainMath(unittest.TestCase):
    def test_converts(self):
        for raw, want in [
            (r"A $\rightarrow$ B", "A → B"),
            (r"A $ \rightarrow $ B", "A → B"),
            (r"A$\to$B$\to$2", "A→B→2"),
            (r"$A \rightarrow B$", "A → B"),
            (r"$$x \le y$$", "x ≤ y"),
            ("$$\nx \\le y\n$$", "\nx ≤ y\n"),
            (r"\(a \neq b\) and \[c \times d\]", "a ≠ b and c × d"),
            (r"$\text{input} \Rightarrow \text{output}$", "input ⇒ output"),
            (r"costs $5 $\rightarrow$ $10", "costs $5 → $10"),
            (r"- $\alpha$" "\n" r"    - $\beta$", "- α\n    - β"),
            ("```\ncode\n```\n" r"$\to$", "```\ncode\n```\n→"),
        ]:
            self.assertEqual(plain_math(raw), want, raw)

    def test_leaves_alone(self):
        for raw in [
            "costs $5 and $10", "echo $HOME/$USER", r"C:\to\in\pi", r"a bare \rightarrow",
            r"$dir\in\$out", r"$root\to\file$", r"$\frac{a}{b}$", r"$x^2$", r"\$\to\$",
            r"\[not a link\]", r"use `$\rightarrow$`", r"use ``a ` $\to$``",
            "`two\n$\\to$ lines`", "```tex\n$\\to$\n```", "~~~\n$\\to$\n~~~",
            "- ```\n  $\\to$\n  ```", "````\n```\n$\\to$\n````", "```\n$\\to$ cut off",
            "text\n\n    $\\to$", "$$a\n\n\\to b$$",
        ]:
            self.assertEqual(plain_math(raw), raw, raw)


class TestDisplay(unittest.TestCase):
    def shown(self, display, text):
        with agent.console.capture() as out:
            display(text)
        return out.get()

    def test_agent_reply(self):
        out = self.shown(agent.show, "A $\\rightarrow$ B\n\n`$\\to$`")
        self.assertIn("A → B", out)
        self.assertIn(r"$\to$", out)

    def test_coder_reply_keeps_code(self):
        fake = coder.Coder.__new__(coder.Coder)
        edit = "a.md\n```md\n<<<<<<< SEARCH\n```\n=======\n$\\le$\n>>>>>>> REPLACE\n```\n"
        text = "A $\\rightarrow$ B\n```tex\n$\\to$\n```\n" + edit + "C $\\ge$ D"
        with patch.object(coder.console, "print") as printed:
            out = self.shown(fake._display, text)
        self.assertIn("A → B", out)
        self.assertIn(r"$\to$", out)
        self.assertIn("C ≥ D", out)
        self.assertIn(r"$\le$", printed.call_args[0][0].code)

    def test_thinking_gist(self):
        self.assertEqual(agent.thinking_gist(r"parse $\to$ run", ""), "parse → run")

    def test_writing_keeps_the_users_latex(self):
        reply = SimpleNamespace(content=r"A $\to$ B")
        for typed, want in (("A then B", "A → B"), (r"A $\to$ B", r"A $\to$ B")):
            with patch("agent_cli.llm.chat", return_value=reply), writing.console.capture() as out:
                writing._run(None, "", typed)
            self.assertIn(want, out.get())


if __name__ == "__main__":
    unittest.main()
