function dummy_matlab_solver(input_csv, output_csv)
    % A realistic simulated physics model written in MATLAB.
    % It includes the cubic trend, interactions, and skewed noise.

    fprintf('   [MATLAB FEA] Booting up... Reading %s\n', input_csv);

    % 1. Read the inputs provided by DigiQual
    try
        df = readtable(input_csv);
    catch
        fprintf('   [MATLAB FEA] Error: Input file not found!\n');
        exit(1);
    end

    % Simulate computation time
    pause(1.0);

    num_rows = height(df);
    signals = NaN(num_rows, 1);

    % Safely check if Roughness exists, otherwise default to 0.0
    has_roughness = ismember('Roughness', df.Properties.VariableNames);

    % 2. Do the "Physics"
    for i = 1:num_rows
        L = df.Length(i);
        angle = df.Angle(i);

        if has_roughness
            roughness = df.Roughness(i);
        else
            roughness = 0.0;
        end

        % A) THE DEAD ZONE (Trigger Graveyard Tracking)
        if (L > 4.0 && L < 6.0) && (abs(angle) > 30)
            signals(i) = NaN;
            continue; % Skip to the next row
        end

        % B) BASE SIGNAL (Cubic Trend + Interaction + Attenuation)
        base_signal = 5.0 + (3.0 * L) - (0.8 * (L^2)) + (0.1 * (L^3)) ...
                      + (angle * 0.1) - (0.05 * L * abs(angle)) - (roughness * 5.0);

        % C) HETEROSCEDASTIC, NON-NORMAL NOISE
        noise_scale = 0.5 + (L * 0.4) + (roughness * 1.0);

        % Generate Gumbel noise using Inverse Transform Sampling (no toolboxes needed!)
        U = rand();
        noise = -noise_scale * log(-log(U));
        noise = noise - (noise_scale * 0.57721);

        signals(i) = base_signal + noise;
    end

    % 3. Save the results back to the hard drive
    df.Signal = signals;
    fprintf('   [MATLAB FEA] Solving complete. Saving to %s\n', output_csv);
    writetable(df, output_csv);

    % CRITICAL: Exit MATLAB so control returns to DigiQual!
    exit;
end
