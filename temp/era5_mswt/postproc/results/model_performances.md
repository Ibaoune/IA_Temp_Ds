# Model Performance Classification

This document provides a quantitative analysis and ranking of the 10 Convolutional Neural Network (CNN) downscaling models evaluated for the Moroccan territory. The scores are derived from the exact spatial averages computed from the NetCDF diagnostic outputs.

## 1. Mean Climate Performance (Bulk Metrics)

The models were evaluated on their ability to capture the general climate state, measured by Mean Bias, Root Mean Square Error (RMSE), and Pearson Correlation (Corr).

**Ranking (Best to Worst):**
1. **CNN10-MSE:** Bias (-0.063), RMSE (0.817), Corr (0.955)
2. **CNN1-MSE:** Bias (-0.063), RMSE (0.820), Corr (0.955)
3. **CNN10-Xiong-Dir:** Bias (-0.110), RMSE (0.835), Corr (0.952)
4. **CNN1-Serifi:** Bias (-0.103), RMSE (0.852), Corr (0.950)
5. **CNN1-Xiong-Dir:** Bias (-0.113), RMSE (0.887), Corr (0.945)
6. **CNN10-Serifi:** Bias (-0.091), RMSE (0.888), Corr (0.945)
7. **CNN1-Xiong-Cont:** Bias (-0.117), RMSE (0.890), Corr (0.945)
8. **CNN10-Xiong-Cont:** Bias (-0.121), RMSE (0.894), Corr (0.944)
9. **CNN-Temp (Baseline):** Bias (-0.453), RMSE (1.096), Corr (0.933)
10. **CNN1-Temp:** Identical to Baseline.

*Conclusion for Means:* The standard MSE-optimized models (`CNN10-MSE`, `CNN1-MSE`) strictly dominate the mean performance. The physics-informed models (`Xiong-Dir`, `Serifi`) follow closely behind, offering a very competitive mean representation. The original baseline (`CNN-Temp`) is by far the worst model.

---

## 2. Extreme Climate Performance (Tails of Distribution)

The models were evaluated on their ability to accurately predict rare/extreme temperature events, measured by the Bias of the 2nd percentile (B02 - cold extremes) and the 98th percentile (B98 - hot extremes). The ranking is based on absolute closeness to 0.

**Cold Extremes (B02) Ranking:**
1. **CNN1-Xiong-Dir:** +0.092
2. **CNN10-Xiong-Cont:** +0.095
3. **CNN1-Xiong-Cont:** +0.100
4. **CNN10-Xiong-Dir:** +0.119
5. **CNN1-MSE:** -0.151
6. **CNN10-MSE:** -0.160
7. **CNN10-Serifi:** +0.220
8. **CNN-Temp (Baseline):** +0.231
9. **CNN1-Temp:** +0.231
10. **CNN1-Serifi:** +0.245

**Hot Extremes (B98) Ranking:**
1. **CNN10-Xiong-Dir:** -0.413
2. **CNN10-Xiong-Cont:** -0.422
3. **CNN1-Serifi:** -0.435
4. **CNN1-Xiong-Cont:** -0.436
5. **CNN10-Serifi:** -0.437
6. **CNN1-Xiong-Dir:** -0.439
7. **CNN1-MSE:** -0.490
8. **CNN10-MSE:** -0.494
9. **CNN-Temp (Baseline):** -0.902
10. **CNN1-Temp:** -0.902

*Conclusion for Extremes:* The Physics-Informed models (PHY-AI), specifically the **Xiong-Directional** and **Xiong-Continuity** variants, drastically outperform the pure MSE models when predicting extreme events. They successfully reduce the cold bias to <0.1°C and halve the hot bias compared to the baseline.

---

## 3. Overall Synthesis & Recommendations

### The "Best of Both Worlds" Model: `CNN10-Xiong-Dir`
The **CNN10-Xiong-Directional** model provides the absolute best compromise. 
- It has the **best prediction for hot extremes** (B98: -0.413).
- It maintains excellent performance on cold extremes (B02: +0.119).
- It is the **best physics-informed model for mean metrics** (RMSE: 0.835, Corr: 0.952), barely trailing behind the pure MSE models.

### The "Mean Climatology" Model: `CNN10-MSE`
If the primary goal of the downscaling is strictly average climate projection without focusing on extreme events, the pure **CNN10-MSE** is statistically the most accurate.

### The Worst Model: `CNN-Temp`
The baseline `CNN-Temp` model severely underestimates hot extremes (nearly -1°C bias) and has significantly higher average errors (RMSE > 1.0) than all newly developed configurations.
