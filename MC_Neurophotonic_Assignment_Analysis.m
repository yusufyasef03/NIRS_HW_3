function results = MC_Neurophotonic_Assignment_Analysis(cfg, transport)
%% GPU PMC Neurophotonic Assignment - Analysis Version
% This file converts stored photon-history summaries into DCS and SCOS
% observables. The DCS/SCOS decorrelation equation uses the stored Y values,
% which are per-layer momentum-transfer summaries, not raw path length.
%
% Teaching organization:
%   1. Start the result tree.
%   2. Compute DCS metrics directly in the main reading path.
%   3. Compute SCOS metrics directly in the main reading path.
%   4. Keep only equation-level helpers below the main analysis.
%
% Student work areas are boxed with STUDENT FILL-IN BLOCK.
% Fill only the blank lines in those blocks. DCS fitting is provided.
%
% MATLAB note:
%   Lines that start with % are comments and will not run.
%% 1. Start the result tree
% Purpose:
%   Collect every analysis output in one result struct while keeping the
%   important setup variables visible.
%
% Variables to notice:
%   cfg       - simulation and analysis settings
%   transport - detected photon weights and path summaries
%   grouped   - transport grouped by detector separation
fprintf('\n------------------------------------------------------------\n');
fprintf('Analyzing stored unpolarized photon transport\n');
fprintf('------------------------------------------------------------\n');
grouped = transport.grouped;
reflectance_per_detector = transport.reflectance_per_detector(:);
results = struct();
results.cfg_snapshot = cfg;
results.detector_rho_mm = cfg.detector_rho_mm(:);
results.reflectance_per_detector = reflectance_per_detector;
baseline_brain_bfi = double(cfg.BFi_baseline_cm2_s(cfg.brain_layer_index));
perturbed_brain_bfi = double(cfg.BFi_perturbed_cm2_s(cfg.brain_layer_index));
results.flow_change_fraction = (perturbed_brain_bfi - baseline_brain_bfi) / max(baseline_brain_bfi, eps);
reference_index = find(abs(cfg.detector_rho_mm(:) - cfg.detector_reference_rho_mm) < 1e-6, 1);
if isempty(reference_index)
    error('Reference detector separation %.1f mm was not found in cfg.detector_rho_mm.', cfg.detector_reference_rho_mm);
end
reference_reflectance = max(reflectance_per_detector(reference_index), eps);
relative_reflectance = reflectance_per_detector / reference_reflectance;
%% 2. DCS analysis
% Purpose:
%   Build DCS correlation curves, fit BFI from selected fitting windows, and
%   estimate sensitivity, CoV, and CNR.
%
% Variables to notice:
%   tau             - DCS correlation delay axis
%   g1_baseline     - electric field correlation at baseline flow
%   g2_baseline     - intensity correlation measured by DCS
%   fit_fraction    - fraction of the decay curve used for one BFI fit
if strcmpi(cfg.mode, 'DCS_ONLY') || strcmpi(cfg.mode, 'BOTH')
    fprintf('\nComputing DCS metrics\n');
    nd = numel(grouped);
    nf = numel(cfg.dcs_fit_fractions);
    tau = cfg.tau_s(:);
    results.DCS = struct();
    results.DCS.tau_s = tau;
    results.DCS.fit_fractions = cfg.dcs_fit_fractions;
    results.DCS.relative_reflectance = relative_reflectance;
    results.DCS.baseline_g1 = zeros(numel(tau), nd);
    results.DCS.perturbed_g1 = zeros(numel(tau), nd);
    results.DCS.baseline_g2 = zeros(numel(tau), nd);
    results.DCS.perturbed_g2 = zeros(numel(tau), nd);
    results.DCS.sensitivity = zeros(nd, nf);
    results.DCS.cov = zeros(nd, nf);
    results.DCS.cnr = zeros(nd, nf);
    results.DCS.fit_bfi_baseline = zeros(nd, nf);
    results.DCS.fit_bfi_perturbed = zeros(nd, nf);
    results.DCS.count_rate_cps = zeros(nd, 1);
    for detector_index = 1:nd
        fprintf('  DCS at separation %2d mm\n', round(grouped(detector_index).rho_mm));
        [~, g1_baseline] = photon_g1_curve(cfg, grouped(detector_index), tau, cfg.BFi_baseline_cm2_s);
        [~, g1_perturbed] = photon_g1_curve(cfg, grouped(detector_index), tau, cfg.BFi_perturbed_cm2_s);
        %% ================= PROVIDED DCS CODE: SIEGERT RELATION ========
        g2_baseline = 1 + cfg.beta_dcs * abs(g1_baseline).^2;
        g2_perturbed = 1 + cfg.beta_dcs * abs(g1_perturbed).^2;
        %% ================= END PROVIDED DCS CODE ======================
        results.DCS.baseline_g1(:, detector_index) = g1_baseline;
        results.DCS.perturbed_g1(:, detector_index) = g1_perturbed;
        results.DCS.baseline_g2(:, detector_index) = g2_baseline;
        results.DCS.perturbed_g2(:, detector_index) = g2_perturbed;
        count_rate_cps = cfg.dcs_modes * cfg.dcs_count_rate_per_mode_at_25mm_cps * relative_reflectance(detector_index) + cfg.dcs_dark_count_cps;
        count_rate_cps = max(count_rate_cps, 1);
        results.DCS.count_rate_cps(detector_index) = count_rate_cps;
        g2_signal = max((g2_baseline(:) - 1) / max(cfg.beta_dcs, eps), 1e-12);
        valid_g2_signal = isfinite(g2_signal) & g2_signal > 0;
        fit_coefficients = polyfit(tau(valid_g2_signal), log(g2_signal(valid_g2_signal)), 1);
        gamma_est = max(-0.5 * fit_coefficients(1), 1);
        sigma_tau = dcs_sigma_tau(cfg, tau, gamma_est, count_rate_cps);
        for fraction_index = 1:nf
            fit_fraction = cfg.dcs_fit_fractions(fraction_index);
            dcs_decay_signal = g2_baseline(:) - 1;
            signal_start = max(dcs_decay_signal(1), eps);
            target_value = signal_start * max(1 - fit_fraction, 0);
            last_fit_index = find(dcs_decay_signal <= target_value, 1);
            if isempty(last_fit_index)
                last_fit_index = numel(dcs_decay_signal);
            end
            fit_mask = false(size(dcs_decay_signal));
            fit_mask(1:last_fit_index) = true;
            tau_fit = tau(fit_mask);
            g2_baseline_fit = g2_baseline(fit_mask);
            g2_perturbed_fit = g2_perturbed(fit_mask);
            %% ================= PROVIDED DCS CODE: BFI FIT =============
            bfi_baseline = solve_bfi_in_log_space(@(trial_bfi) sum((1 + cfg.beta_dcs * abs(semi_infinite_g1(cfg, grouped(detector_index).rho_mm, tau_fit, trial_bfi)).^2 - g2_baseline_fit) .^ 2), cfg);
            bfi_perturbed = solve_bfi_in_log_space(@(trial_bfi) sum((1 + cfg.beta_dcs * abs(semi_infinite_g1(cfg, grouped(detector_index).rho_mm, tau_fit, trial_bfi)).^2 - g2_perturbed_fit) .^ 2), cfg);
            results.DCS.fit_bfi_baseline(detector_index, fraction_index) = bfi_baseline;
            results.DCS.fit_bfi_perturbed(detector_index, fraction_index) = bfi_perturbed;
            results.DCS.sensitivity(detector_index, fraction_index) = ((bfi_perturbed - bfi_baseline) / max(bfi_baseline, eps)) / max(results.flow_change_fraction, eps);
            %% ================= END PROVIDED DCS CODE ==================
            noisy_fits = zeros(cfg.dcs_realizations, 1);
            for realization_index = 1:cfg.dcs_realizations
                noisy_g2 = g2_baseline + sigma_tau .* randn(size(g2_baseline));
                noisy_g2 = max(noisy_g2, 1 + 1e-12);
                noisy_g2_fit = noisy_g2(fit_mask);
                %% ================= PROVIDED DCS CODE: NOISY REFIT =====
                noisy_fits(realization_index) = solve_bfi_in_log_space(@(trial_bfi) sum((1 + cfg.beta_dcs * abs(semi_infinite_g1(cfg, grouped(detector_index).rho_mm, tau_fit, trial_bfi)).^2 - noisy_g2_fit) .^ 2), cfg);
                %% ================= END PROVIDED DCS CODE ==============
            end
            results.DCS.cov(detector_index, fraction_index) = coefficient_of_variation_from_noisy_fits(noisy_fits, cfg);
            if ~isfinite(results.DCS.cov(detector_index, fraction_index)) || results.DCS.cov(detector_index, fraction_index) <= 0
                results.DCS.cnr(detector_index, fraction_index) = NaN;
            else
                results.DCS.cnr(detector_index, fraction_index) = results.DCS.sensitivity(detector_index, fraction_index) / results.DCS.cov(detector_index, fraction_index);
            end
        end
    end
end
%% 3. SCOS analysis
% Purpose:
%   Integrate field correlations over camera exposure time, fit BFI from
%   speckle contrast, and estimate sensitivity, CoV, and CNR.
%
% Variables to notice:
%   tau_scos        - delay axis used for the exposure integral
%   exposure_s      - one camera exposure time
%   sp_ratio        - speckle-to-pixel ratio used in the noise model
%   kf2_baseline    - baseline fundamental speckle contrast squared
if strcmpi(cfg.mode, 'SCOS_ONLY') || strcmpi(cfg.mode, 'BOTH')
    fprintf('\nComputing SCOS metrics\n');
    nd = numel(grouped);
    ne = numel(cfg.exposure_s);
    ns = numel(cfg.scos_sp_ratios);
    tau_scos = cfg.tau_scos_s(:);
    texp = cfg.exposure_s(:);
    results.SCOS = struct();
    results.SCOS.exposure_s = texp;
    results.SCOS.sp_ratios = cfg.scos_sp_ratios;
    results.SCOS.relative_reflectance = relative_reflectance;
    results.SCOS.baseline_kf2 = zeros(ne, nd);
    results.SCOS.perturbed_kf2 = zeros(ne, nd);
    results.SCOS.fit_bfi_baseline = zeros(nd, ne);
    results.SCOS.fit_bfi_perturbed = zeros(nd, ne);
    results.SCOS.sensitivity = zeros(nd, ne);
    results.SCOS.cov = zeros(nd, ne);
    results.SCOS.cnr = zeros(nd, ne);
    results.SCOS.best_exposure_idx = zeros(nd, 1);
    results.SCOS.best_exposure_s = zeros(nd, 1);
    results.SCOS.best_sensitivity = zeros(nd, 1);
    results.SCOS.best_cov = zeros(nd, 1);
    results.SCOS.best_cnr = zeros(nd, 1);
    results.SCOS.sp_ratio_cnr_15mm = zeros(ns, 1);
    results.SCOS.sp_ratio_cnr_30mm = zeros(ns, 1);
    baseline_g1_cache = cell(nd, 1);
    perturbed_g1_cache = cell(nd, 1);
    for detector_index = 1:nd
        fprintf('  SCOS at separation %2d mm\n', round(grouped(detector_index).rho_mm));
        [~, baseline_g1_cache{detector_index}] = photon_g1_curve(cfg, grouped(detector_index), tau_scos, cfg.BFi_baseline_cm2_s);
        [~, perturbed_g1_cache{detector_index}] = photon_g1_curve(cfg, grouped(detector_index), tau_scos, cfg.BFi_perturbed_cm2_s);
        for exposure_index = 1:ne
            exposure_s = texp(exposure_index);
            sp_ratio = cfg.scos_default_sp_ratio;
            kf2_baseline = scos_kf2_from_g1(tau_scos, baseline_g1_cache{detector_index}, exposure_s, cfg.beta_scos);
            kf2_perturbed = scos_kf2_from_g1(tau_scos, perturbed_g1_cache{detector_index}, exposure_s, cfg.beta_scos);
            bfi_baseline = fit_bfi_from_scos(cfg, grouped(detector_index).rho_mm, exposure_s, kf2_baseline, cfg.beta_scos);
            bfi_perturbed = fit_bfi_from_scos(cfg, grouped(detector_index).rho_mm, exposure_s, kf2_perturbed, cfg.beta_scos);
            %% ================= STUDENT FILL-IN BLOCK SCOS-1A ================
            % Compute sensitivity as fractional change in fitted BFI over true change
            sensitivity = ((bfi_perturbed - bfi_baseline) / max(bfi_baseline, eps)) / max(results.flow_change_fraction, eps);
            %% ================= END STUDENT FILL-IN BLOCK SCOS-1A ============
            noise = scos_noise_model(cfg, relative_reflectance(detector_index), exposure_s, sp_ratio, kf2_baseline);
            noisy_fits = zeros(cfg.scos_realizations, 1);
            for realization_index = 1:cfg.scos_realizations
                noisy_kf2 = max(kf2_baseline + noise.sigma_kf2 * randn(), 1e-12);
                noisy_fits(realization_index) = fit_bfi_from_scos(cfg, grouped(detector_index).rho_mm, exposure_s, noisy_kf2, cfg.beta_scos);
            end
            cov_value = coefficient_of_variation_from_noisy_fits(noisy_fits, cfg);
            results.SCOS.baseline_kf2(exposure_index, detector_index) = kf2_baseline;
            results.SCOS.perturbed_kf2(exposure_index, detector_index) = kf2_perturbed;
            results.SCOS.fit_bfi_baseline(detector_index, exposure_index) = bfi_baseline;
            results.SCOS.fit_bfi_perturbed(detector_index, exposure_index) = bfi_perturbed;
            results.SCOS.sensitivity(detector_index, exposure_index) = sensitivity;
            results.SCOS.cov(detector_index, exposure_index) = cov_value;
            if ~isfinite(cov_value) || cov_value <= 0
                results.SCOS.cnr(detector_index, exposure_index) = NaN;
            else
                results.SCOS.cnr(detector_index, exposure_index) = sensitivity / cov_value;
            end
        end
        [results.SCOS.best_cnr(detector_index), results.SCOS.best_exposure_idx(detector_index)] = max(results.SCOS.cnr(detector_index, :));
        results.SCOS.best_exposure_s(detector_index) = texp(results.SCOS.best_exposure_idx(detector_index));
        results.SCOS.best_sensitivity(detector_index) = results.SCOS.sensitivity(detector_index, results.SCOS.best_exposure_idx(detector_index));
        results.SCOS.best_cov(detector_index) = results.SCOS.cov(detector_index, results.SCOS.best_exposure_idx(detector_index));
    end
    idx_15 = find(abs(cfg.detector_rho_mm - 15) < 1e-6, 1);
    idx_30 = find(abs(cfg.detector_rho_mm - 30) < 1e-6, 1);
    if ~isempty(idx_15)
        for sp_index = 1:ns
            best_cnr = -inf;
            for exposure_index = 1:ne
                exposure_s = texp(exposure_index);
                sp_ratio = cfg.scos_sp_ratios(sp_index);
                kf2_baseline = scos_kf2_from_g1(tau_scos, baseline_g1_cache{idx_15}, exposure_s, cfg.beta_scos);
                kf2_perturbed = scos_kf2_from_g1(tau_scos, perturbed_g1_cache{idx_15}, exposure_s, cfg.beta_scos);
                bfi_baseline = fit_bfi_from_scos(cfg, grouped(idx_15).rho_mm, exposure_s, kf2_baseline, cfg.beta_scos);
                bfi_perturbed = fit_bfi_from_scos(cfg, grouped(idx_15).rho_mm, exposure_s, kf2_perturbed, cfg.beta_scos);
                %% ================= STUDENT FILL-IN BLOCK SCOS-1B ================
                % Calculate sensitivity for 15mm separation using the same formula
                sensitivity = ((bfi_perturbed - bfi_baseline) / max(bfi_baseline, eps)) / max(results.flow_change_fraction, eps);
                %% ================= END STUDENT FILL-IN BLOCK SCOS-1B ============
                noise = scos_noise_model(cfg, relative_reflectance(idx_15), exposure_s, sp_ratio, kf2_baseline);
                noisy_fits = zeros(cfg.scos_realizations, 1);
                for realization_index = 1:cfg.scos_realizations
                    noisy_kf2 = max(kf2_baseline + noise.sigma_kf2 * randn(), 1e-12);
                    noisy_fits(realization_index) = fit_bfi_from_scos(cfg, grouped(idx_15).rho_mm, exposure_s, noisy_kf2, cfg.beta_scos);
                end
                cov_value = coefficient_of_variation_from_noisy_fits(noisy_fits, cfg);
                if isfinite(cov_value) && cov_value > 0
                    best_cnr = max(best_cnr, sensitivity / cov_value);
                end
            end
            results.SCOS.sp_ratio_cnr_15mm(sp_index) = best_cnr;
        end
    end
    if ~isempty(idx_30)
        for sp_index = 1:ns
            best_cnr = -inf;
            for exposure_index = 1:ne
                exposure_s = texp(exposure_index);
                sp_ratio = cfg.scos_sp_ratios(sp_index);
                kf2_baseline = scos_kf2_from_g1(tau_scos, baseline_g1_cache{idx_30}, exposure_s, cfg.beta_scos);
                kf2_perturbed = scos_kf2_from_g1(tau_scos, perturbed_g1_cache{idx_30}, exposure_s, cfg.beta_scos);
                bfi_baseline = fit_bfi_from_scos(cfg, grouped(idx_30).rho_mm, exposure_s, kf2_baseline, cfg.beta_scos);
                bfi_perturbed = fit_bfi_from_scos(cfg, grouped(idx_30).rho_mm, exposure_s, kf2_perturbed, cfg.beta_scos);
                %% ================= STUDENT FILL-IN BLOCK SCOS-1C ================
                %%Calculate sensitivity for 30mm separation
                sensitivity = ((bfi_perturbed - bfi_baseline) / max(bfi_baseline, eps)) / max(results.flow_change_fraction, eps);
                %% ================= END STUDENT FILL-IN BLOCK SCOS-1C ============
                noise = scos_noise_model(cfg, relative_reflectance(idx_30), exposure_s, sp_ratio, kf2_baseline);
                noisy_fits = zeros(cfg.scos_realizations, 1);
                for realization_index = 1:cfg.scos_realizations
                    noisy_kf2 = max(kf2_baseline + noise.sigma_kf2 * randn(), 1e-12);
                    noisy_fits(realization_index) = fit_bfi_from_scos(cfg, grouped(idx_30).rho_mm, exposure_s, noisy_kf2, cfg.beta_scos);
                end
                cov_value = coefficient_of_variation_from_noisy_fits(noisy_fits, cfg);
                if isfinite(cov_value) && cov_value > 0
                    best_cnr = max(best_cnr, sensitivity / cov_value);
                end
            end
            results.SCOS.sp_ratio_cnr_30mm(sp_index) = best_cnr;
        end
    end
end
end % <--- CRITICAL FIX: Bu 'end' ana MC_Neurophotonic_Assignment_Analysis fonksiyonunu kapatır.

%% Helper: Photon data to field correlation
function [G1, g1] = photon_g1_curve(cfg, group, tau, BFi_cm2_s) 
if group.n < 1 
    G1 = zeros(size(tau)); 
    g1 = zeros(size(tau)); 
    return; 
end 

%% ================= STUDENT FILL-IN BLOCK SHARED-1 =================
w = group.w(:);                                     % Extract photon weights as a vector
Y = group.Y;                                       % Extract pathlength/momentum matrix per layer
BFi_vector = BFi_cm2_s(:);                         % Convert BFI vector into column format
k0_vac_cm_inv = 2 * pi / (cfg.lambda_nm * 1e-7);   % Calculate wavenumber in vacuum (1/cm)

% Equation 11: Calculate photon-specific decorrelation rates (1D vector)
dynamic_rate = 2 * (k0_vac_cm_inv^2) * (double(cfg.n_tissue)^2) * (Y * BFi_vector);

% Equation 12 & 13: Compute unnormalized and normalized G1/g1 curves
G1 = weighted_exponential_curve(tau, dynamic_rate, w, 100000);

% FIX: Cast group.n to double to prevent integer combination error
g1 = G1 / max(sum(w) / max(double(group.n), 1), eps);      % Normalize by zero-delay average weight
%% ================= END STUDENT FILL-IN BLOCK SHARED-1 ============
end 

%% Helper: Memory-safe photon average
function G1 = weighted_exponential_curve(tau, dynamic_rate, weights, chunk_size)
tau = tau(:);
dynamic_rate = dynamic_rate(:);
weights = weights(:);
G1 = zeros(numel(tau), 1);
n_photons = numel(dynamic_rate); 
for start_index = 1:chunk_size:n_photons
    stop_index = min(start_index + chunk_size - 1, n_photons);
    photon_slice = start_index:stop_index;
    decay_block = exp(-dynamic_rate(photon_slice) * tau(:).'); 
    G1 = G1 + decay_block' * weights(photon_slice); 
end 
G1 = G1 / max(n_photons, 1); 
end 

%% Helper: Semi-infinite field correlation
function g1 = semi_infinite_g1(cfg, rho_mm, tau, bfi) 
mua_cm_inv = cfg.dcs_fit_mua_cm_inv; 
musp_cm_inv = cfg.dcs_fit_musp_cm_inv; 
rho_cm = rho_mm / 10; 
n_tissue = double(cfg.n_tissue); 
k0_vac_cm_inv = 2 * pi / (cfg.lambda_nm * 1e-7); 
l_star_cm = 1 / musp_cm_inv; 
Reff = -1.440 * n_tissue ^ -2 + 0.710 * n_tissue ^ -1 + 0.668 + 0.0636 * n_tissue; 
zb_cm = (2 / (3 * musp_cm_inv)) * ((1 + Reff) / (1 - Reff)); 
r1_cm = sqrt(rho_cm ^ 2 + l_star_cm ^ 2); 
rb_cm = sqrt(rho_cm ^ 2 + (l_star_cm + 2 * zb_cm) ^ 2); 
K0_cm_inv = sqrt(3 * mua_cm_inv * musp_cm_inv); 
K_cm_inv = sqrt(3 * mua_cm_inv * musp_cm_inv + 6 * (k0_vac_cm_inv ^ 2) * (n_tissue ^ 2) * (musp_cm_inv ^ 2) * bfi .* tau); 
numerator = rb_cm .* exp(-K_cm_inv .* r1_cm) - r1_cm .* exp(-K_cm_inv .* rb_cm); 
denominator = rb_cm * exp(-K0_cm_inv * r1_cm) - r1_cm * exp(-K0_cm_inv * rb_cm); 
g1 = numerator ./ max(denominator, eps); 
end 

%% Helper: DCS noise model
function sigma_tau = dcs_sigma_tau(cfg, tau, gamma_est, count_rate_cps) 
bin_width = cfg.dcs_bin_width_s; 
sample_time = cfg.sample_time_s; 
nbar = max(count_rate_cps * bin_width, 1e-12); 
exp_2gamma_bin = exp(-2 * gamma_est * bin_width); 
exp_2gamma_tau = exp(-2 * gamma_est * tau(:)); 
exp_gamma_tau = exp(-gamma_est * tau(:)); 
term_one = cfg.beta_dcs ^ 2 .* ((1 + exp_2gamma_bin) .* (1 + exp_2gamma_tau) + 2 * (tau(:) / bin_width) .* (1 - exp_2gamma_bin) .* exp_2gamma_tau) ./ max(1 - exp_2gamma_bin, eps); 
term_two = 2 * (nbar ^ -1) * cfg.beta_dcs .* (1 + exp_2gamma_tau); 
term_three = (nbar ^ -2) .* (1 + cfg.beta_dcs .* exp_gamma_tau); 
sigma_tau = sqrt((bin_width / sample_time) .* (term_one + term_two + term_three)); 
end 

%% Helper: SCOS exposure integral
function kf2 = scos_kf2_from_g1(tau, g1, exposure_s, beta) 
%% ================= STUDENT FILL-IN BLOCK SCOS-INTEGRAL =============
tau = tau(:);
g1_squared = abs(g1(:)).^2;  
sample_mask = tau < exposure_s;
tau_use = tau(sample_mask);
g1_squared_use = g1_squared(sample_mask);
if isempty(tau_use) || tau_use(1) > 0
    tau_use = [0; tau_use];
    g1_squared_use = [1; g1_squared_use];
end
if tau_use(end) < exposure_s
    endpoint_value = interp1(tau, g1_squared, exposure_s, 'linear', 0);
    tau_use = [tau_use; exposure_s];
    g1_squared_use = [g1_squared_use; max(endpoint_value, 0)];
end
integrand = g1_squared_use .* (1 - tau_use / exposure_s);
kf2 = (2 * beta / exposure_s) * trapz(tau_use, integrand);
%% ================= END STUDENT FILL-IN BLOCK SCOS-INTEGRAL ========
end 

%% Helper: SCOS BFI fit
function bfi = fit_bfi_from_scos(cfg, rho_mm, exposure_s, measured_kf2, beta) 
bfi = solve_bfi_in_log_space(@(trial_bfi) (scos_kf2_from_g1(cfg.tau_scos_s(:), semi_infinite_g1(cfg, rho_mm, cfg.tau_scos_s(:), trial_bfi), exposure_s, beta) - measured_kf2) ^ 2, cfg); 
end 

%% Helper: Log-space BFI solver
function bfi = solve_bfi_in_log_space(objective_in_bfi, cfg) 
log10_bfi_min = log10(double(cfg.bfi_fit_bounds_cm2_s(1))); 
log10_bfi_max = log10(double(cfg.bfi_fit_bounds_cm2_s(2))); 
objective_in_log10_bfi = @(trial_log10_bfi) objective_in_bfi(10 .^ trial_log10_bfi); 
solver_options = optimset('Display', 'off', 'TolX', 1e-4); 
log10_bfi = fminbnd(objective_in_log10_bfi, log10_bfi_min, log10_bfi_max, solver_options); 
bfi = 10 .^ log10_bfi; 
if isempty(bfi) || ~isfinite(bfi) 
    bfi = double(cfg.BFi_baseline_cm2_s(cfg.brain_layer_index)); 
end 
end 

%% Helper: CoV from noisy BFI fits
function cov_value = coefficient_of_variation_from_noisy_fits(noisy_fits, cfg) 
fit_min = double(cfg.bfi_fit_bounds_cm2_s(1)); 
fit_max = double(cfg.bfi_fit_bounds_cm2_s(2)); 
log10_fit_values = log10(max(double(noisy_fits(:)), realmin)); 
lower_bound_hits = abs(log10_fit_values - log10(fit_min)) <= cfg.bfi_fit_boundary_log10_tolerance; 
upper_bound_hits = abs(log10_fit_values - log10(fit_max)) <= cfg.bfi_fit_boundary_log10_tolerance; 
if mean(lower_bound_hits | upper_bound_hits) > cfg.max_fraction_noisy_fits_on_solver_boundary 
    cov_value = NaN; 
    return; 
end 
cov_value = std(noisy_fits) / max(mean(noisy_fits), eps); 
end 

%% Helper: SCOS noise model
function noise = scos_noise_model(cfg, relative_reflectance, exposure_s, sp_ratio, kf2) 
frame_rate_hz = min(cfg.scos_frame_rate_hz, 1 / max(exposure_s, eps)); 
frames_per_sample = max(floor(frame_rate_hz * cfg.sample_time_s), 1); 
filled_pixels = max(min(cfg.scos_pixels, cfg.scos_bundle_modes * sp_ratio ^ 2), 1); 
independent_observations_per_frame = max(min(cfg.scos_bundle_modes, cfg.scos_pixels / max(sp_ratio ^ 2, eps)), 1); 
photoelectrons_per_second = cfg.scos_count_rate_per_mode_at_25mm_cps * relative_reflectance * cfg.scos_bundle_modes * cfg.scos_qe; 
photoelectrons_per_frame = max(photoelectrons_per_second * exposure_s, 1e-12); 
mean_photoelectrons_per_pixel = max(photoelectrons_per_frame / filled_pixels, 1e-12); 
shot_contrast_sq = 1 / mean_photoelectrons_per_pixel; 
read_contrast_sq = (cfg.scos_read_noise_e ^ 2) / (mean_photoelectrons_per_pixel ^ 2); 
measured_k2 = max(kf2 + shot_contrast_sq + read_contrast_sq, 1e-12); 
%% ================= CRITICAL FIX: COMPLETED THE TRUNCATED FUNCTION CODE =================
nio_total = max(independent_observations_per_frame * frames_per_sample, 1); 
noise = struct(); 
noise.frame_rate_hz = frame_rate_hz; 
noise.frames_per_sample = frames_per_sample; 
noise.filled_pixels = filled_pixels; 
noise.independent_observations_per_frame = independent_observations_per_frame; 
noise.photoelectrons_per_frame = photoelectrons_per_frame; 
noise.mean_photoelectrons_per_pixel = mean_photoelectrons_per_pixel; 
noise.shot_contrast_sq = shot_contrast_sq; 
noise.read_contrast_sq = read_contrast_sq; 
noise.sigma_kf2 = measured_k2 * sqrt(2 / nio_total); 
end