r"""Show simple LaTeX math as plain text: $\rightarrow$ becomes →.

For display only, never for what is saved, run or sent back. A span is changed
only if every command in it is known, so code, paths, prices, shell variables
and math that has no plain-text form stay as written. Line count is preserved.
"""
import re

_SYMBOLS = dict(pair.split("=", 1) for pair in """
rightarrow=→ to=→ longrightarrow=⟶ Rightarrow=⇒ Longrightarrow=⟹ implies=⟹
leftarrow=← gets=← longleftarrow=⟵ Leftarrow=⇐ Longleftarrow=⟸
leftrightarrow=↔ Leftrightarrow=⇔ iff=⟺ mapsto=↦
uparrow=↑ downarrow=↓ nearrow=↗ searrow=↘
le=≤ leq=≤ ge=≥ geq=≥ ne=≠ neq=≠ ll=≪ gg=≫ approx=≈ equiv=≡ sim=∼ propto=∝
times=× div=÷ pm=± mp=∓ cdot=· circ=∘ bullet=•
dots=… ldots=… cdots=⋯ checkmark=✓
infty=∞ in=∈ notin=∉ subset=⊂ subseteq=⊆ cup=∪ cap=∩ emptyset=∅
forall=∀ exists=∃ neg=¬ land=∧ wedge=∧ lor=∨ vee=∨ sum=∑ prod=∏ partial=∂ nabla=∇
alpha=α beta=β gamma=γ delta=δ epsilon=ε varepsilon=ε zeta=ζ eta=η theta=θ
iota=ι kappa=κ lambda=λ mu=μ nu=ν xi=ξ pi=π rho=ρ sigma=σ tau=τ phi=φ varphi=φ
chi=χ psi=ψ omega=ω Gamma=Γ Delta=Δ Theta=Θ Lambda=Λ Pi=Π Sigma=Σ Phi=Φ Psi=Ψ Omega=Ω
log=log ln=ln min=min max=max
""".split())
_SYMBOLS.update({"quad": " ", "qquad": "  ", ",": " ", ";": " ", " ": " ", "%": "%"})

# Fenced, inline and indented code (an indented list item is prose).
_CODE = (r"(?ms:^[ \t>]*(?:(?:[-*+]|\d+[.)])[ \t]+)?(?P<fence>`{3,}(?=[^`\n]*$)|~{3,})"
         r".*?(?:^[ \t>]*(?P=fence)[`~]*[ \t\r]*$|\Z))"
         r"|(?<!`)(?P<ticks>`+)(?!`)"
         r"(?:[^\n]|\n(?![ \t]*(?:\n|```|~~~|[-*+][ \t])))+?(?<!`)(?P=ticks)(?!`)"
         r"|(?m:^(?: {4}|\t)(?![ \t]*(?:[-*+]|\d+[.)])[ \t])[^\n]*)")
_MATH = (r"(?<![\\$])\$\$(?P<block>[^$`]{1,500})\$\$"
         r"|\\\[(?P<bracket>[^$`]{1,500}?)\\\]"
         r"|\\\((?P<paren>[^$`\n]{1,300}?)\\\)"
         # only commands, e.g. $\rightarrow$ or $ \to $: may touch the text around it
         r"|(?<![\\$])\$(?P<symbols>[ \t]*(?:\\[A-Za-z]+[ \t]*)+)\$"
         # anything else: pandoc's rule, so "$5 and $10" and $HOME are not math
         r"|(?<![\\$\w])\$(?P<inline>(?=\S)[^$`\n]{1,300}(?<=\S))\$(?![\w$])")
_SPAN_RE = re.compile(f"(?P<code>{_CODE})|{_MATH}")
_WORDS_RE = re.compile(r"\\(?:text|mathrm|mathbf|mathit|textbf|operatorname)\{([^{}\\$]*)\}")
_COMMAND_RE = re.compile(r"\\([A-Za-z]+|[^A-Za-z\n])")


def _convert(match) -> str:
    if match.group("code") is not None:
        return match.group()
    body = next(v for k in ("block", "bracket", "paren", "symbols", "inline")
                for v in [match.group(k)] if v is not None)
    if "\n\n" in body:
        return match.group()
    plain, words = _WORDS_RE.subn(r"\1", body)
    commands = _COMMAND_RE.findall(plain)
    if not (words or commands) or any(c not in _SYMBOLS for c in commands):
        return match.group()
    return _COMMAND_RE.sub(lambda m: _SYMBOLS[m.group(1)], plain).strip(" \t")


def plain_math(text: str) -> str:
    return _SPAN_RE.sub(_convert, text) if "\\" in text else text
