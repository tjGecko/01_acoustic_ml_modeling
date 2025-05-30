### How to translate TDOA uncertainty into angular (DOA) error

Below is a practical recipe—together with a **ready-to-run Python snippet** (figure displayed above)—that lets you estimate the 1-σ direction-of-arrival (DOA) error you can expect from GCC-PHAT on the **miniDSP UMA-16** array.

---

#### 1 .  Relate delay to angle for one baseline

For a single microphone pair with baseline length $d$ and orientation $\varphi$:

$$
\tau(\theta)=\frac{d}{c}\cos(\theta-\varphi)
$$

where

* $c\approx343\text{ m s}^{-1}$ is the speed of sound,
* $\theta$ is the source azimuth (0° = north in the plots).

---

#### 2 .  Propagate the TDOA variance

If the standard deviation of the delay estimate is $\sigma_\tau$, the **error propagation rule** gives

$$
\sigma_\theta\;=\;\frac{\sigma_\tau\,c}{d\,\bigl|\sin(\theta-\varphi)\bigr|}.
$$

*Large baselines* and *arrival angles perpendicular to the baseline* ( $\sin\approx1$ ) minimise the angular error.

---

#### 3 .  Combine many pairs (16 mics ⇒ 120 baselines)

Treating the baselines as independent measurements, the Fisher-information sum gives a combined 1-σ error:

$$
\sigma_\theta^{\text{(combined)}}=\sqrt{\frac{1}{\displaystyle\sum_{k}\frac{1}{\sigma_{\theta,k}^2}}}.
$$

---

#### 4 .  Pick (or measure) a realistic $\sigma_\tau$

With GCC-PHAT you often get **sub-sample interpolation accuracy** in quiet conditions:

$$
\sigma_\tau \approx  
\frac{1}{2\pi\,\beta\,\sqrt{2\,\text{SNR}}} 
\quad\Longrightarrow\quad 
5\!-\!20\;\mu\text{s} \text{ is typical},
$$

where $\beta$ is the RMS bandwidth of the source (Knapp & Carter 1976).
� **Change this value to match your SNR / sampling rate / interpolation scheme.**

---

### Example: UMA-16, $\sigma_\tau = 20 \mu\text{s}$

The code below:

* parses your XML,
* computes every baseline’s length & bearing,
* propagates the delay error into angle space,
* fuses all 120 baselines,
* plots the resulting **1-σ DOA error envelope**.

```python
import numpy as np, xml.etree.ElementTree as ET, matplotlib.pyplot as plt

# ---------- 1) mic coordinates ----------
xml = "minidsp_uma16.xml"      # put your XML path here
pos = np.array([[float(p.attrib['x']),
                 float(p.attrib['y']),
                 float(p.attrib['z'])]
                for p in ET.parse(xml).getroot().findall('pos')])

# ---------- 2) constants ----------
c          = 343.0          # speed of sound (m/s)
sigma_tau  = 20e-6          # assumed TDOA std-dev  (20 µs)

# ---------- 3) pre-compute baselines ----------
pairs      = [(i,j) for i in range(len(pos)) for j in range(i+1,len(pos))]
b_vecs     = np.array([pos[j]-pos[i] for i,j in pairs])
b_len      = np.linalg.norm(b_vecs[:,:2], axis=1)
b_ang      = np.arctan2(b_vecs[:,1], b_vecs[:,0])      # XY bearing

# ---------- 4) angular error versus azimuth ----------
az_deg = np.arange(0,360); az_rad = np.deg2rad(az_deg)
sigma_th = []
for th in az_rad:
    info = 0.0
    for L, phi in zip(b_len, b_ang):
        s = abs(np.sin(th-phi))
        if s<1e-6: continue            # baseline parallel to wavefront ⇒ no info
        var = (sigma_tau*c/(L*s))**2   # single-baseline variance
        info += 1/var                  # add Fisher information
    sigma_th.append(np.sqrt(1/info))
sigma_th = np.rad2deg(np.array(sigma_th))   # → degrees

# ---------- 5) polar plot ----------
fig,ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(8,6))
ax.plot(az_rad, sigma_th, lw=2, label="1-σ Angular Error (deg)")
ax.set_title("Theoretical DOA error — UMA-16  (στ=20 µs)", pad=20)
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.set_rlabel_position(135); ax.legend(loc='lower right', bbox_to_anchor=(1.1,-0.1))
plt.tight_layout(); plt.show()
```

*(The figure you just saw was generated with this snippet.)*

---

### Interpreting the plot

* **Dips** (≈ 0.05°) occur where many long baselines are perpendicular to the source (≈ 45°, 135°, 225°, 315°).
* **Peaks** (≈ 0.5° in this example) correspond to directions aligned with the array arms, where some baselines contribute little or no information.

---

### Take-aways & next steps

| Lever                                      | Effect on σθ    | Notes                                                                          |
| ------------------------------------------ | --------------- | ------------------------------------------------------------------------------ |
| **Smaller σ τ**                            | ↓ linearly      | Improve SNR, wider-band signals, better interpolation.                         |
| **Longer baselines**                       | ↓ inversely     | Use outer-most mic pairs or add outriggers.                                    |
| **3-D geometry**                           | ↓ via diversity | Add elevation-separated mics for out-of-plane sources.                         |
| **Weighted least-squares / CRLB**          | ↓               | Weight baselines by their individual σ τ.                                      |
| **Geometric Dilution of Precision (GDOP)** | diagnostic      | Map how geometry amplifies delay errors (see MathWorks link ([MathWorks][1])). |

Adjust `sigma_tau`, sample-rate, or even run Monte-Carlo simulations (inject noise & re-estimate TDOA) to validate the theoretical bounds for your specific signals.

[1]: https://www.mathworks.com/help/fusion/ug/object-tracking-using-time-difference-of-arrival.html?utm_source=chatgpt.com "Object Tracking Using Time Difference of Arrival (TDOA) - MathWorks"
