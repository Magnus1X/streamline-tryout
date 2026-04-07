import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("📈 Linear Regression Interactive Visualizer")

# -----------------------------
# 1. DATA GENERATION
# -----------------------------
st.sidebar.header("Controls")

n = st.sidebar.slider("Number of points", 20, 200, 50)
noise = st.sidebar.slider("Noise level", 0.0, 10.0, 2.0)
add_outliers = st.sidebar.checkbox("Add outliers")

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, n)
true_m, true_b = 2, 5
y = true_m * X + true_b + np.random.randn(n) * noise

if add_outliers:
    y[:5] += 20  # inject outliers

# -----------------------------
# 2. LINE PARAMETERS
# -----------------------------
st.sidebar.subheader("Line Controls")
m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)

y_pred = m * X + b

# -----------------------------
# 3. MSE FUNCTION
# -----------------------------
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = compute_mse(y, y_pred)

# -----------------------------
# 4. DATA + LINE PLOT
# -----------------------------
st.subheader("📊 Data & Line Fit")

fig, ax = plt.subplots()
ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color="red", label="Model Line")

# Residual lines
for i in range(len(X)):
    ax.plot([X[i], X[i]], [y[i], y_pred[i]], linestyle="dashed")

ax.legend()
st.pyplot(fig)

st.write(f"### 📉 MSE: {mse:.2f}")

# -----------------------------
# 5. LOSS LANDSCAPE (CONTOUR)
# -----------------------------
st.subheader("🌄 Loss Landscape")

m_vals = np.linspace(-1, 4, 50)
b_vals = np.linspace(0, 10, 50)

M, B = np.meshgrid(m_vals, b_vals)
Z = np.zeros_like(M)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Z[i, j] = compute_mse(y, M[i, j]*X + B[i, j])

fig2 = go.Figure(data=go.Contour(
    z=Z,
    x=m_vals,
    y=b_vals,
    contours_coloring='heatmap'
))

fig2.add_trace(go.Scatter(
    x=[m],
    y=[b],
    mode='markers',
    marker=dict(size=10, color='red'),
    name="Current (m,b)"
))

st.plotly_chart(fig2)

# -----------------------------
# 6. GRADIENT DESCENT
# -----------------------------
st.subheader("⚙️ Gradient Descent")

lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
iterations = st.slider("Iterations", 10, 200, 50)

def gradient_descent(X, y, lr, iterations):
    m, b = 0, 0
    m_list, b_list, loss_list = [], [], []

    n = len(X)

    for _ in range(iterations):
        y_pred = m * X + b

        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        m -= lr * dm
        b -= lr * db

        loss = compute_mse(y, y_pred)

        m_list.append(m)
        b_list.append(b)
        loss_list.append(loss)

    return m_list, b_list, loss_list

m_list, b_list, loss_list = gradient_descent(X, y, lr, iterations)

# Plot loss curve
fig3, ax3 = plt.subplots()
ax3.plot(loss_list)
ax3.set_title("Loss vs Iterations")
st.pyplot(fig3)

# -----------------------------
# 7. GD PATH ON CONTOUR
# -----------------------------
st.subheader("🧭 Optimization Path")

fig4 = go.Figure(data=go.Contour(
    z=Z,
    x=m_vals,
    y=b_vals,
    contours_coloring='heatmap'
))

fig4.add_trace(go.Scatter(
    x=m_list,
    y=b_list,
    mode='lines+markers',
    name="GD Path"
))

st.plotly_chart(fig4)

# -----------------------------
# 8. EXPLANATION
# -----------------------------
st.markdown("""
### 🧠 What to Observe:
- Adjust slope & intercept → see fit change
- Watch MSE decrease as line improves
- Gradient Descent moves toward minimum
- Large learning rate may diverge
- Outliers distort the best-fit line
""")