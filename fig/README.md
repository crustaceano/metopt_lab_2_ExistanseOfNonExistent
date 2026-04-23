# Figures Folder

Store all exported plots for the report here.

Recommended naming:
- `exp2_2_*.png`
- `exp2_3_*.png`
- `exp2_4_*.png`
- `exp2_5_*.png`
- `exp2_6_*.png`
- `exp7_*.png`

In notebooks, save plots with:

```python
import os
os.makedirs("fig", exist_ok=True)
plt.savefig("fig/exp2_3_grad_vs_iter.png", dpi=200, bbox_inches="tight")
```
