"""V6 feature battery (~85 candidates, v6_ prefix) for financial-stress challenge.

Design rationale (from iter14 residual autopsy + v5 failure analysis):
- v1-v5 exhausted mask x ratio/slope interactions (v5: 0/79 kept by greedy OOF).
- Autopsy findings driving v6:
  1. HVC segment has 57% higher error than MVC (0.321 vs 0.205 LL) -> hard-regime flags.
  2. ARPU U-shape: both top and bottom arpu deciles are hard (0.268/0.253 vs 0.20 mid)
     -> peer-relative (arpu-decile-conditioned) normalizations.
  3. Network-diversity columns (senders/agents/merchants/recipients/banks/companies)
     have ZERO engineered usage in v1-v5, yet diversity TRENDS correlate with target
     (dep_agents slope +0.105, senders slope +0.073, wd_agents slope -0.066).
     -> full diversity-dynamics family ("liquidity scramble" signature).
  4. Inflow cliffs: sharp m1 drops in received/deposit -> worst LL pockets.
  5. 75% of worst-decile errors are false negatives (missed stress) -> hidden-stress flags.
  6. Balance volatility (high CV) -> worst quintile.

ALL FEATURES ARE UNSUPERVISED (no target use). Quantile cutoffs / peer stats are
fit on TRAIN and applied to test (no transduction, no leakage).
"""
import numpy as np
import pandas as pd

EPS = 1e-9


def _monthly(df, pattern):
    cols = [c for c in df.columns if _match_monthly(c, pattern)]
    return df[cols].values.astype(float), cols


def _match_monthly(col, pattern):
    # pattern like 'm{}_paybill_companies' -> matches m1..m6
    return any(col == pattern.format(i) for i in range(1, 7))


def add_features_v6(tr, te):
    """Returns (dict_tr, dict_te) of new feature name -> array. No target use."""
    new_tr, new_te = {}, {}

    def add(name, v_tr, v_te):
        if name in tr.columns or name in new_tr:
            raise RuntimeError(f"name collision: {name}")
        a_tr = np.nan_to_num(np.asarray(v_tr, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        a_te = np.nan_to_num(np.asarray(v_te, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        new_tr[name] = a_tr
        new_te[name] = a_te

    def monthly(df, metric):
        cols = [f"m{i}_{metric}" for i in range(1, 7)]
        return df[cols].values.astype(float)

    t = np.arange(6, dtype=float)
    tc = t - t.mean()
    tss = (tc ** 2).sum()

    def slope(a):
        return (a * tc).sum(axis=1) / tss

    def cv(a):
        mu = np.abs(a.mean(axis=1))
        return a.std(axis=1) / np.maximum(mu, 1e-9)

    # ============ shared base series ============
    b = monthly(tr, "daily_avg_bal"); bte = monthly(te, "daily_avg_bal")
    chans = ["paybill", "merchantpay", "transfer_from_bank", "mm_send",
             "deposit", "withdraw", "received"]
    tv = {c: monthly(tr, f"{c}_total_value") for c in chans}
    tvte = {c: monthly(te, f"{c}_total_value") for c in chans}
    vol = {c: monthly(tr, f"{c}_volume") for c in chans}
    volte = {c: monthly(te, f"{c}_volume") for c in chans}
    hi = {c: monthly(tr, f"{c}_highest_amount") for c in chans}
    hite = {c: monthly(te, f"{c}_highest_amount") for c in chans}
    inflow = tv["deposit"] + tv["received"]
    inflowte = tvte["deposit"] + tvte["received"]
    outflow = tv["withdraw"] + tv["mm_send"] + tv["paybill"] + tv["merchantpay"]
    outflowte = tvte["withdraw"] + tvte["mm_send"] + tvte["paybill"] + tvte["merchantpay"]
    act = tr["x_90_d_activity_rate"].values.astype(float)
    actte = te["x_90_d_activity_rate"].values.astype(float)
    age = tr["age"].values.astype(float); agete = te["age"].values.astype(float)
    arpu = tr["arpu"].values.astype(float); arpute = te["arpu"].values.astype(float)
    seg = tr["segment"].astype(str).values; segte = te["segment"].astype(str).values
    is_hvc = (seg == "HVC").astype(float); is_hvcte = (segte == "HVC").astype(float)
    is_lvc = (seg == "LVC").astype(float); is_lvcte = (segte == "LVC").astype(float)

    # ============ FAMILY A: network-diversity dynamics (fresh) ============
    div_map = {
        "pbco": "paybill_companies",
        "merch": "merchantpay_merchants",
        "bank": "transfer_from_bank_banks",
        "recip": "mm_send_recipients",
        "sender": "received_senders",
        "depag": "deposit_agents",
        "wdag": "withdraw_agents",
    }
    div, divte = {}, {}
    for short, metric in div_map.items():
        div[short] = monthly(tr, metric)
        divte[short] = monthly(te, metric)
        d, dte = div[short], divte[short]
        med_hist = np.median(d[:, 1:], axis=1)
        med_histte = np.median(dte[:, 1:], axis=1)
        # A1: trend slope of diversity
        add(f"v6_div_{short}_slope", slope(d), slope(dte))
        # A2: m1 vs historical median ratio
        add(f"v6_div_{short}_m1vshist", d[:, 0] / np.maximum(med_hist, 1.0),
            dte[:, 0] / np.maximum(med_histte, 1.0))

    # A4: value per counterparty (avg ticket per relationship), m1 and trend
    vpc_pairs = [("received", "sender"), ("deposit", "depag"), ("withdraw", "wdag"),
                 ("mm_send", "recip"), ("paybill", "pbco"), ("merchantpay", "merch"),
                 ("transfer_from_bank", "bank")]
    for ch, dshort in vpc_pairs:
        v, vte_ = tv[ch], tvte[ch]
        d, dte_ = div[dshort], divte[dshort]
        vpc = v / np.maximum(d, 1.0)
        vpcte = vte_ / np.maximum(dte_, 1.0)
        add(f"v6_vpc_{ch}_m1", vpc[:, 0], vpcte[:, 0])
        add(f"v6_vpc_{ch}_slope", slope(vpc), slope(vpcte))

    # A5: liquidity-scramble signature: deposit-agent expansion vs withdrawal-agent contraction
    scr = slope(div["depag"]) - slope(div["wdag"])
    scrte = slope(divte["depag"]) - slope(divte["wdag"])
    add("v6_div_scramble", scr, scrte)
    add("v6_div_scramble_x_hvc", is_hvc * scr, is_hvcte * scrte)
    # sender expansion while received value flat/falling
    snd_up = (slope(div["sender"]) > 0).astype(float)
    snd_upte = (slope(divte["sender"]) > 0).astype(float)
    recv_down = (slope(tv["received"]) < 0).astype(float)
    recv_downte = (slope(tvte["received"]) < 0).astype(float)
    add("v6_div_senders_up_recv_down", snd_up * recv_down, snd_upte * recv_downte)

    # A6: total network breadth and its concentration
    breadth = sum(div.values())
    breadthte = sum(divte.values())
    add("v6_div_breadth_m1", breadth[:, 0], breadthte[:, 0])
    add("v6_div_breadth_slope", slope(breadth), slope(breadthte))
    add("v6_div_breadth_cv", cv(breadth), cv(breadthte))

    # ============ FAMILY B: peer-relative / arpu-conditioned (fresh) ============
    # arpu deciles fit on train, applied to test
    q = np.quantile(arpu, np.linspace(0, 1, 11))
    q[0], q[-1] = -np.inf, np.inf
    dec_tr = np.clip(np.digitize(arpu, q[1:-1]), 0, 9)
    dec_te = np.clip(np.digitize(arpute, q[1:-1]), 0, 9)
    add("v6_arpu_decile", dec_tr.astype(float), dec_te.astype(float))
    # per-decile median/IQR of key behaviors from TRAIN only
    peer_feats = {}
    for name, arr, arrte in [("bal_m1", b[:, 0], bte[:, 0]),
                             ("outflow_m1", outflow[:, 0], outflowte[:, 0]),
                             ("inflow_m1", inflow[:, 0], inflowte[:, 0]),
                             ("act", act, actte)]:
        med = np.array([np.median(arr[dec_tr == d]) for d in range(10)])
        iqr = np.array([np.subtract(*np.percentile(arr[dec_tr == d], [75, 25])) for d in range(10)])
        iqr = np.maximum(iqr, 1e-9)
        z_tr = (arr - med[dec_tr]) / iqr[dec_tr]
        z_te = (arrte - med[dec_te]) / iqr[dec_te]
        peer_feats[name] = (z_tr, z_te)
        add(f"v6_peer_{name}_z", np.clip(z_tr, -5, 5), np.clip(z_te, -5, 5))
    # B2: rank mismatch — balance rank vs arpu rank (percentile points)
    r_bal = pd.qcut(pd.Series(b[:, 0]), 100, labels=False, duplicates="drop").values / 100.0
    r_arpu = pd.qcut(pd.Series(arpu), 100, labels=False, duplicates="drop").values / 100.0
    # map test through train quantiles for consistency
    qb = np.quantile(b[:, 0], np.linspace(0, 1, 101)); qb[0], qb[-1] = -np.inf, np.inf
    qa = np.quantile(arpu, np.linspace(0, 1, 101)); qa[0], qa[-1] = -np.inf, np.inf
    r_balte = np.clip(np.digitize(bte[:, 0], qb[1:-1]), 0, 99) / 100.0
    r_arpute = np.clip(np.digitize(arpute, qa[1:-1]), 0, 99) / 100.0
    add("v6_rank_mismatch_bal_arpu", r_bal - r_arpu, r_balte - r_arpute)
    # B3: outflow relative to arpu-decile peer median
    peer_out_med = np.array([np.median(outflow[:, 0][dec_tr == d]) for d in range(10)])
    add("v6_outflow_vs_peer", outflow[:, 0] / np.maximum(peer_out_med[dec_tr], 1.0),
        outflowte[:, 0] / np.maximum(peer_out_med[dec_te], 1.0))

    # ============ FAMILY C: time-series shape (fresh) ============
    def entropy(a):
        # normalized Shannon entropy of the 6-month distribution
        s = np.abs(a).sum(axis=1, keepdims=True)
        p_ = np.abs(a) / np.maximum(s, 1e-9)
        p_ = np.clip(p_, 1e-9, 1)
        return -(p_ * np.log(p_)).sum(axis=1) / np.log(6)

    def longest_run(a):
        # longest monotonic run (in steps) of the 6-month series
        runs = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            ss = np.sign(np.diff(a[i]))
            mx, c = 1, 1
            for k in range(1, len(ss)):
                if ss[k] == ss[k - 1] and ss[k] != 0:
                    c += 1
                else:
                    c = 1
                mx = max(mx, c)
            runs[i] = mx
        return runs

    def n_dir_changes(a):
        s = np.sign(np.diff(a, axis=1))
        # count sign changes ignoring zeros
        ch = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            nz = s[i][s[i] != 0]
            ch[i] = np.sum(nz[1:] != nz[:-1]) if len(nz) > 1 else 0
        return ch

    for sname, arr, arrte in [("bal", b, bte), ("inflow", inflow, inflowte),
                              ("outflow", outflow, outflowte),
                              ("net", inflow - outflow, inflowte - outflowte)]:
        add(f"v6_shape_{sname}_entropy", entropy(arr), entropy(arrte))
        add(f"v6_shape_{sname}_dirchanges", n_dir_changes(arr), n_dir_changes(arrte))
        add(f"v6_shape_{sname}_longrun", longest_run(arr), longest_run(arrte))
        # time since peak (0 = m1 is the peak)
        pk = np.argmax(arr, axis=1).astype(float)
        pkte = np.argmax(arrte, axis=1).astype(float)
        add(f"v6_shape_{sname}_t_since_peak", pk, pkte)
    # C5: second-half vs first-half (m1-m3 vs m4-m6)
    add("v6_shape_inflow_late_early",
        inflow[:, :3].sum(axis=1) / np.maximum(inflow[:, 3:].sum(axis=1), 1.0),
        inflowte[:, :3].sum(axis=1) / np.maximum(inflowte[:, 3:].sum(axis=1), 1.0))
    add("v6_shape_outflow_late_early",
        outflow[:, :3].sum(axis=1) / np.maximum(outflow[:, 3:].sum(axis=1), 1.0),
        outflowte[:, :3].sum(axis=1) / np.maximum(outflowte[:, 3:].sum(axis=1), 1.0))

    # ============ FAMILY D: ticket-size / transaction structure (fresh) ============
    for ch in chans:
        v, vte_ = tv[ch], tvte[ch]
        n, nte_ = vol[ch], volte[ch]
        h, hte_ = hi[ch], hite[ch]
        avg_ticket = v / np.maximum(n, 1.0)
        avg_tickette = vte_ / np.maximum(nte_, 1.0)
        add(f"v6_ticket_{ch}_m1", avg_ticket[:, 0], avg_tickette[:, 0])
        add(f"v6_ticket_{ch}_slope", slope(avg_ticket), slope(avg_tickette))
        conc = h / np.maximum(v, 1.0)  # largest tx share of channel value
        conce = hte_ / np.maximum(vte_, 1.0)
        add(f"v6_conc_{ch}_m1", conc[:, 0], conce[:, 0])
    # D4: volume-value divergence (counts vs money moving differently)
    tot_vol = sum(vol.values()); tot_volte = sum(volte.values())
    tot_val = sum(tv.values()); tot_valte = sum(tvte.values())
    add("v6_volval_diverge",
        slope(tot_vol) - slope(tot_val) / np.maximum(np.abs(tot_val).mean(axis=1), 1.0),
        slope(tot_volte) - slope(tot_valte) / np.maximum(np.abs(tot_valte).mean(axis=1), 1.0))

    # ============ FAMILY E: cliff detectors (fresh binary formulation) ============
    def cliff(a, ate, thresh=0.3):
        med = np.median(a[:, 1:], axis=1)
        medte = np.median(ate[:, 1:], axis=1)
        f = ((a[:, 0] < thresh * med) & (med > 0)).astype(float)
        fte = ((ate[:, 0] < thresh * medte) & (medte > 0)).astype(float)
        return f, fte

    for cname, arr, arrte in [("received", tv["received"], tvte["received"]),
                              ("deposit", tv["deposit"], tvte["deposit"]),
                              ("withdraw", tv["withdraw"], tvte["withdraw"]),
                              ("paybill", tv["paybill"], tvte["paybill"]),
                              ("inflow", inflow, inflowte),
                              ("bal", b, bte)]:
        f, fte = cliff(arr, arrte)
        add(f"v6_cliff_{cname}", f, fte)
        add(f"v6_cliff_{cname}_x_hvc", is_hvc * f, is_hvcte * fte)
    # E2: sudden stop — m1 == 0 after nonzero history
    for cname, arr, arrte in [("received", tv["received"], tvte["received"]),
                              ("deposit", tv["deposit"], tvte["deposit"])]:
        s = ((arr[:, 0] == 0) & (arr[:, 1:].sum(axis=1) > 0)).astype(float)
        ste = ((arrte[:, 0] == 0) & (arrte[:, 1:].sum(axis=1) > 0)).astype(float)
        add(f"v6_suddenstop_{cname}", s, ste)

    # ============ FAMILY F: channel substitution & bank channel (fresh) ============
    # paybill share of (paybill+merchantpay): migration between bill channels
    pb = tv["paybill"]; pbte = tvte["paybill"]
    mp = tv["merchantpay"]; mpte = tvte["merchantpay"]
    pb_share = pb / np.maximum(pb + mp, 1.0)
    pb_sharete = pbte / np.maximum(pbte + mpte, 1.0)
    add("v6_chmig_pbshare_slope", slope(pb_share), slope(pb_sharete))
    # cash vs digital: (withdraw+deposit)/(mm_send+received+1)
    cash = tv["withdraw"] + tv["deposit"]
    cashte = tvte["withdraw"] + tvte["deposit"]
    digi = tv["mm_send"] + tv["received"]
    digite = tvte["mm_send"] + tvte["received"]
    cash_share = cash / np.maximum(cash + digi, 1.0)
    cash_sharete = cashte / np.maximum(cashte + digite, 1.0)
    add("v6_cashshare_m1", cash_share[:, 0], cash_sharete[:, 0])
    add("v6_cashshare_slope", slope(cash_share), slope(cash_sharete))
    # bank top-up behavior: transfer_from_bank trend and share of inflow
    tfb = tv["transfer_from_bank"]; tfbte = tvte["transfer_from_bank"]
    add("v6_banktopup_slope", slope(tfb), slope(tfbte))
    add("v6_banktopup_share_m1", tfb[:, 0] / np.maximum(inflow[:, 0], 1.0),
        tfbte[:, 0] / np.maximum(inflowte[:, 0], 1.0))
    # net bank flow: bank-in minus cash-out
    nbf = tfb - tv["withdraw"]
    nbfte = tfbte - tvte["withdraw"]
    add("v6_netbankflow_m1", nbf[:, 0], nbfte[:, 0])
    add("v6_netbankflow_slope", slope(nbf), slope(nbfte))

    # ============ FAMILY H: acute-shock detectors (from Top-25 August solution) ============
    # distress is an acute transactional event (Markov diff only 0.8pp), not a
    # persistent trait -> second differences / collapse counts / runway
    deficit = outflow - inflow
    deficitte = outflowte - inflowte
    d1, d2, d3 = deficit[:, 0], deficit[:, 1], deficit[:, 2]
    d1te, d2te, d3te = deficitte[:, 0], deficitte[:, 1], deficitte[:, 2]
    add("v6_shock_deficit_accel", (d1 - d2) - (d2 - d3),
        (d1te - d2te) - (d2te - d3te))
    add("v6_shock_bal_jerk",
        b[:, 0] - 3 * b[:, 1] + 3 * b[:, 2] - b[:, 3],
        bte[:, 0] - 3 * bte[:, 1] + 3 * bte[:, 2] - bte[:, 3])
    # acute weekly runway: (bal_m1+1) / (max(0, net weekly burn)+1)
    burn = np.maximum(outflow[:, 0] - inflow[:, 0], 0) / 4.0
    burnte = np.maximum(outflowte[:, 0] - inflowte[:, 0], 0) / 4.0
    add("v6_shock_runway_wk", (b[:, 0] + 1) / (burn + 1),
        (bte[:, 0] + 1) / (burnte + 1))
    # lumpy inflow: single largest receipt / total inflow (windfall dependence)
    lump = hi["received"][:, 0] / np.maximum(inflow[:, 0], 1.0)
    lumpte = hite["received"][:, 0] / np.maximum(inflowte[:, 0], 1.0)
    add("v6_shock_lumpy_inflow", lump, lumpte)
    # per-channel dormancy: months since last nonzero volume (0 = active in m1)
    for ch, v_, vte_ in [("withdraw", vol["withdraw"], volte["withdraw"]),
                         ("deposit", vol["deposit"], volte["deposit"]),
                         ("received", vol["received"], volte["received"]),
                         ("paybill", vol["paybill"], volte["paybill"])]:
        nz = (v_ > 0).astype(int)
        nzte = (vte_ > 0).astype(int)
        # argmax on reversed gives distance from m1; handle all-zero -> 6
        dor = np.full(v_.shape[0], 6.0)
        dorte = np.full(vte_.shape[0], 6.0)
        for i in range(6):
            m = (nz[:, i] == 1) & (dor == 6)
            dor[m] = i
            mte = (nzte[:, i] == 1) & (dorte == 6)
            dorte[mte] = i
        add(f"v6_shock_dormant_{ch}", dor, dorte)
    # inflow-collapse count: key inflow channels dropping <0.7x q1->q2 style
    # (m1 vs mean(m2,m3) per channel, count of collapses)
    coll = np.zeros(tr.shape[0])
    collte = np.zeros(te.shape[0])
    for ch in ["deposit", "received", "transfer_from_bank"]:
        v_, vte_ = tv[ch], tvte[ch]
        hist = v_[:, 1:4].mean(axis=1)
        histte = vte_[:, 1:4].mean(axis=1)
        coll += ((v_[:, 0] < 0.7 * hist) & (hist > 0)).astype(float)
        collte += ((vte_[:, 0] < 0.7 * histte) & (histte > 0)).astype(float)
    add("v6_shock_inflow_collapse_ct", coll, collte)

    # ============ FAMILY G: hard-regime flags (fresh) ============
    bal_cv = cv(b)
    bal_cvte = cv(bte)
    bal_cv_med = np.median(bal_cv)
    z_tot = sum((div[s] == 0).sum(axis=1) for s in div)
    z_totte = sum((divte[s] == 0).sum(axis=1) for s in divte)
    add("v6_hard_hvc_volatile", (is_hvc * (bal_cv > bal_cv_med)).astype(float),
        (is_hvcte * (bal_cvte > bal_cv_med)).astype(float))
    # HVC-relative thresholds (HVC users are active, so global medians degenerate)
    hvc_z_med = np.median(z_tot[seg == "HVC"]) if (seg == "HVC").any() else np.median(z_tot)
    hvc_act_med = np.median(act[seg == "HVC"]) if (seg == "HVC").any() else 0.3
    add("v6_hard_hvc_sparse", (is_hvc * (z_tot > hvc_z_med)).astype(float),
        (is_hvcte * (z_totte > hvc_z_med)).astype(float))
    add("v6_hard_hvc_lowact", (is_hvc * (act < hvc_act_med)).astype(float),
        (is_hvcte * (actte < hvc_act_med)).astype(float))
    # hidden stress: top-half balance + inflow cliff
    bal_med = np.median(b[:, 0])
    f_inflow, f_inflowte = cliff(inflow, inflowte)
    add("v6_hidden_okbal_inflowcliff",
        ((b[:, 0] > bal_med) & (f_inflow > 0)).astype(float),
        ((bte[:, 0] > bal_med) & (f_inflowte > 0)).astype(float))
    # high arpu + low activity (affluent disengaged)
    arpu_p80 = np.quantile(arpu, 0.8)
    add("v6_hidden_rich_inactive",
        ((arpu > arpu_p80) & (act < 0.3)).astype(float),
        ((arpute > arpu_p80) & (actte < 0.3)).astype(float))

    return new_tr, new_te


if __name__ == "__main__":
    import os
    DATA = os.path.expanduser("~/workspace/financial-stress-prediction-challenge-2026-10-09/data")
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    dtr, dte = add_features_v6(tr, te)
    print(f"v6 battery: {len(dtr)} candidates")
    bad = []
    for k, v in dtr.items():
        if not np.all(np.isfinite(v)):
            bad.append((k, "tr non-finite"))
    for k, v in dte.items():
        if not np.all(np.isfinite(v)):
            bad.append((k, "te non-finite"))
    # constant check
    for k, v in dtr.items():
        if np.std(v) == 0:
            bad.append((k, "tr constant"))
    print("BAD:", bad if bad else "none — all finite, non-constant")
    # train/test distribution sanity for a few
    for k in ["v6_div_depag_slope", "v6_peer_bal_m1_z", "v6_rank_mismatch_bal_arpu",
              "v6_shape_inflow_entropy", "v6_ticket_withdraw_m1", "v6_cliff_received",
              "v6_div_scramble", "v6_banktopup_slope"]:
        a, b_ = dtr[k], dte[k]
        print(f"{k}: tr_mean={a.mean():.4f} te_mean={b_.mean():.4f} "
              f"tr_std={a.std():.4f} te_std={b_.std():.4f}")
