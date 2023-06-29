#pragma once

namespace fv1d {

State primToCons(State &q) {
  State res;
  res[IR] = q[IR];
  res[IU] = q[IR]*q[IU];
  res[IV] = q[IR]*q[IV];

  real_t Ek = 0.5 * (res[IU]*res[IU] + res[IV]*res[IV]) / q[IR];
  res[IE] = (Ek + q[IP] / (gamma0-1.0));
  return res;
}

State consToPrim(State &u) {
  State res;
  res[IR] = u[IR];
  res[IU] = u[IU] / u[IR];
  res[IV] = u[IV] / u[IR];

  real_t Ek = 0.5 * res[IR] * (res[IU]*res[IU] + res[IV]*res[IV]);
  res[IP] = (u[IE] - Ek) * (gamma0-1.0);
  return res; 
}

void consToPrim(Array &U, Array &Q) {
  for (int j=0; j < Nty; ++j)
    for (int i=0; i < Ntx; ++i)
      Q[j][i] = consToPrim(U[j][i]);
}

void primToCons(Array &Q, Array &U) {
  for (int j=0; j < Nty; ++j)
    for (int i=0; i < Ntx; ++i)
      U[j][i] = primToCons(Q[j][i]);
}

real_t speed_of_sound(State &q) {
  return std::sqrt(q[IP] * gamma0 / q[IR]);
}

State& operator+=(State &a, State b) {
  for (int i=0; i < Nfields; ++i)
    a[i] += b[i];
  return a;
}

State& operator-=(State &a, State b) {
  for (int i=0; i < Nfields; ++i)
    a[i] -= b[i];
  return a;
}

State operator*(const State &a, real_t q) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]*q;
  return res;
}

State operator/(const State &a, real_t q) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]/q;
  return res;
}

State operator*(real_t q, const State &a) {
  return a*q;
}

State operator+(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]+b[i];
  return res;
}

State operator-(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]-b[i];
  return res;
}

State swap_component(State &q, IDir dir) {
  if (dir == IX)
    return q;
  else
    return {q[IR], q[IV], q[IU], q[IP]};
}

State compute_flux(State &q) {
  const real_t Ek = 0.5 * q[IR] * (q[IU] * q[IU] + q[IV] * q[IV]);
  const real_t E = (q[IP] / (gamma0-1.0) + Ek);

  State fout {
    q[IR]*q[IU],
    q[IR]*q[IU]*q[IU] + q[IP],
    q[IR]*q[IU]*q[IV],
    (q[IP] + E) * q[IU]
  };

  return fout;
}
}