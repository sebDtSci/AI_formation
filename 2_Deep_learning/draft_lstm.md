## Formules
### LSTM :
$$
f_t = \sigma\left(W_f \cdot [h_{t-1}, x_t] + b_f\right)
$$
$$
i_t = \sigma\left(W_i \cdot [h_{t-1}, x_t] + b_i\right)
$$
$$
\tilde{C}_t = \tanh\left(W_C \cdot [h_{t-1}, x_t] + b_C\right)
$$
$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$
$$
o_t = \sigma\left(W_o \cdot [h_{t-1}, x_t] + b_o\right)
$$
$$
h_t = o_t \cdot \tanh(C_t)
$$
### GRU :
$$
z_t = \sigma\left(W_z \cdot [h_{t-1}, x_t] + b_z\right)
$$
$$
r_t = \sigma\left(W_r \cdot [h_{t-1}, x_t] + b_r\right)
$$
$$
\tilde{h}_t = \tanh\left(W_h \cdot [r_t \cdot h_{t-1}, x_t] + b_h\right)
$$
$$
h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
$$

