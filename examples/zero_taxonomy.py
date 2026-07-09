"""The same visible 0 hiding four different inner states."""
from uncollapsed import UncollapsedField

for f in [
    UncollapsedField.void(source="void        "),
    UncollapsedField(0.18, 0.18, source="calm centre  "),
    UncollapsedField(0.90, 0.90, source="contradiction"),
    UncollapsedField(0.85, 0.35, source="presence lean"),
]:
    m = f.mass()
    print(f"{f.source} | icon={f.icon().glyph()} | belief={m.belief:.2f} "
          f"disbelief={m.disbelief:.2f} conflict={m.conflict:.2f} void={m.voidness:.2f} "
          f"| collapse -> {f.collapse().result.value}")
