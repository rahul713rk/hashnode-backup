---
title: "Understanding Root Mean Squared Logarithmic Error ( RMSLE )"
datePublished: Wed May 07 2025 09:49:58 GMT+0000 (Coordinated Universal Time)
cuid: cmadrbwom000009jjewxy8pfa
slug: understanding-root-mean-squared-logarithmic-error-rmsle
tags: rmsle

---

# Explaination

## A. What is RMSLE?

RMSLE is an evaluation metric used for regression problems. It measures the ratio-based difference between predicted and actual values rather than absolute differences. RMSLE is especially useful when:

* You care more about **relative** error than absolute error.
    
* Your target variable spans multiple orders of magnitude.
    
* You want to **penalize under-predictions more** than over-predictions (as the log curve is asymmetric).
    

---

## B. In More Simple Terms

### 🧠 Imagine this situation:

You’re building a model that predicts **house prices**. Some houses are worth ₹1 lakh, others ₹1 crore. That’s a *huge* range!

Now, say your model predicts:

* For a ₹1 lakh house → ₹90,000 (so you’re off by ₹10,000)
    
* For a ₹1 crore house → ₹91 lakh (off by ₹9 lakh)
    

Which is the bigger mistake?

If you just use normal error (like Mean Squared Error), it says the ₹9 lakh mistake is worse — **but in reality**, being off by 10% is the same in both cases.

---

### 📏 That's where RMSLE comes in

Instead of comparing raw values, RMSLE looks at the **ratio** between the predicted and actual value using **logarithms**.

Why logs?

* Logs **compress large numbers** — a crore and a lakh become numbers like 12 and 5.
    
* Logs turn **ratios into differences**, which helps us compare things fairly.
    
* It **penalizes underestimates more** than overestimates — which can be good when under-predicting is more harmful (e.g., underestimating demand or risk).
    

---

# **Mathematical Derivation**

Let’s start from a more common metric and derive our way to RMSLE.

$$y_i = \text{Actual value} $$

 $$\hat{y}_i = \text{Predicted value}$$

### **A. MSE (Mean Squared Error)**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

It penalizes absolute error heavily.

### **B. MSLE (Mean Squared Logarithmic Error)**

Instead of direct differences, we take the difference in logarithmic space:

$$\text{MSLE} = \frac{1}{n} \sum_{i=1}^{n} \left( \log_e(1 + y_i) - \log_e(1 + \hat{y}_i) \right)^2$$

* The addition of 1 avoids issues with log(0).
    
* It reduces the effect of large values and focuses more on **percentage** differences.
    

### **C. RMSLE (Root Mean Squared Logarithmic Error)**

Take the square root of MSLE:

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log_e(1 + y_i) - \log_e(1 + \hat{y}_i) \right)^2}$$