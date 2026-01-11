function perm = ...
    upscalePermeabilityPeriodicEL(Gp, bcp, dp_scale, psolver, fluid, rock, L, ringSize)
% Upscale permeability on a periodic grid using the Extended Local method.
%
% SYNOPSIS:
%   perm = upscalePermeabilityPeriodicExtendedLocal(Gp, bcp, dp_scale, psolver, fluid, rock, L, ringSize)
%
% DESCRIPTION:
%   This function calculates the effective permeability tensor by solving 
%   local flow problems on an extended domain (Gp) and extracting fluxes 
%   within a central core defined by the ringSize.
%
% PARAMETERS:
%   Gp       - Periodic grid object (extended domain)
%   bcp      - Boundary conditions structure
%   dp_scale - Magnitude of the pressure drop to apply
%   psolver  - Handle to the pressure solver (e.g., @incompTPFA)
%   fluid    - Fluid properties (must be single-phase)
%   rock     - Rock properties (porosity, permeability)
%   L        - Physical dimensions [Lx, Ly, Lz] of the extended grid
%   ringSize - Number of cells in the buffer zone [rx, ry, rz]
%
% RETURNS:
%   perm    - 3x3 upscaled permeability tensor
%   state   - Final solution state from the last simulation run
%   vmat    - 3x3 matrix of average velocities
%   dp_mat  - 3x3 matrix of average pressure gradients

    %% 1. Grid and Domain Dimensions
    % Gp is the "Extended" grid. We calculate the dimensions of the "Core" block.
    dims = Gp.cartDims;
    d    = Gp.griddim;
    
    % Dimensions of the entire extended block
    ecgx = dims(1); ecgy = dims(2); ecgz = dims(3);
    
    % Dimensions of the core block (subtracting the rings)
    cgx = ecgx - ringSize(1);
    cgy = ecgy - ringSize(2);
    cgz = ecgz - ringSize(3);

    % Define coordinate ranges for the core block
    xmin = ringSize(1)/2 + 1;  xmax = ecgx - ringSize(1)/2;
    ymin = ringSize(2)/2 + 1;  ymax = ecgy - ringSize(2)/2;
    zmin = ringSize(3)/2 + 1;  zmax = ecgz - ringSize(3)/2;

    %% 2. Index Mapping (Find Core Faces)
    % We need the faces on the inlet and outlet of the core block to measure 
    % the pressure drop and flux specifically across the central region.
    
    % --- X-Direction Faces ---
    mXInlet = []; mXOutlet = [];
    for z = zmin:zmax
        for y = ymin:ymax
            mXInlet(end+1)  = xmin - 1 + (y-1)*ecgx + (z-1)*ecgx*ecgy;
            mXOutlet(end+1) = xmax     + (y-1)*ecgx + (z-1)*ecgx*ecgy;
        end
    end
    XInletIndex  = zeros(size(mXInlet));
    XOutletIndex = zeros(size(mXInlet));
    for i = 1:numel(mXInlet)
        XInletIndex(i)  = find(Gp.faces.neighbors(:, 1) == mXInlet(i) & Gp.faces.neighbors(:, 2) == mXInlet(i)+1);
        XOutletIndex(i) = find(Gp.faces.neighbors(:, 1) == mXOutlet(i) & Gp.faces.neighbors(:, 2) == mXOutlet(i)+1);
    end

    % --- Y-Direction Faces ---
    mYInlet = []; mYOutlet = [];
    for z = zmin:zmax
        for x = xmin:xmax
            mYInlet(end+1)  = x + (ymin-2)*ecgx + (z-1)*ecgx*ecgy;
            mYOutlet(end+1) = x + (ymax-1)*ecgx + (z-1)*ecgx*ecgy;
        end
    end
    YInletIndex  = zeros(size(mYInlet));
    YOutletIndex = zeros(size(mYInlet));
    for i = 1:numel(mYInlet)
        YInletIndex(i)  = find(Gp.faces.neighbors(:, 1) == mYInlet(i) & Gp.faces.neighbors(:, 2) == mYInlet(i)+ecgx);
        YOutletIndex(i) = find(Gp.faces.neighbors(:, 1) == mYOutlet(i) & Gp.faces.neighbors(:, 2) == mYOutlet(i)+ecgx);
    end

    % --- Z-Direction Faces ---
    mZInlet = []; mZOutlet = [];
    for x = xmin:xmax
        for y = ymin:ymax
            mZInlet(end+1)  = x + (y-1)*ecgx + (zmin-2)*ecgx*ecgy;
            mZOutlet(end+1) = x + (y-1)*ecgx + (zmax-1)*ecgx*ecgy;
        end
    end
    ZInletIndex  = zeros(size(mZInlet));
    ZOutletIndex = zeros(size(mZInlet));
    for i = 1:numel(mZInlet)
        ZInletIndex(i)  = find(Gp.faces.neighbors(:, 1) == mZInlet(i) & Gp.faces.neighbors(:, 2) == mZInlet(i)+ecgx*ecgy);
        ZOutletIndex(i) = find(Gp.faces.neighbors(:, 1) == mZOutlet(i) & Gp.faces.neighbors(:, 2) == mZOutlet(i)+ecgx*ecgy);
    end

    %% 3. Solve Local Flow Problems
    vmat    = nan(d);
    dp_mat  = dp_scale * eye(d);
    bcp_new = bcp;
    state   = initResSol(Gp, 100*barsa, 1);
    
    % Store individual states for each directional solve
    states = cell(d, 1);
    dp_core = zeros(d, 1);

    for i = 1:d
        % Set pressure drops for current direction i
        for j = 1:d
            bcp_new.value(bcp.tags == j) = dp_mat(j, i);
        end
        
        % Solve flow on the extended grid
        state = psolver(state, Gp, fluid, bcp_new, rock);
        states{i} = state;

        % Calculate internal core pressure drop for the primary direction
        if i == 1
            dp_core(1) = sum(state.facePressure(XOutletIndex) - state.facePressure(XInletIndex));
        elseif i == 2
            dp_core(2) = sum(state.facePressure(YOutletIndex) - state.facePressure(YInletIndex));
        elseif i == 3
            dp_core(3) = sum(state.facePressure(ZOutletIndex) - state.facePressure(ZInletIndex));
        end

        % --- Calculate Volumetric Average Flux across the Core ---
        % Sum fluxes through all internal face layers in the core
        flux_x = 0;
        for n = 1:numel(XInletIndex)
            flux_x = flux_x - sum(state.flux(XInletIndex(n)+1 : XOutletIndex(n)));
        end
        
        flux_y = 0;
        for n = 1:numel(YInletIndex)
            yind = YInletIndex(n)+ecgx : ecgx : YOutletIndex(n);
            flux_y = flux_y - sum(state.flux(yind));
        end
        
        flux_z = 0;
        for n = 1:numel(ZInletIndex)
            zind = ZInletIndex(n)+(ecgx*ecgy) : (ecgx*ecgy) : ZOutletIndex(n);
            flux_z = flux_z - sum(state.flux(zind));
        end

        % Normalize by the number of cells in the core (Volumetric Average)
        vmat(1, i) = flux_x / (cgx*cgy*cgz);
        vmat(2, i) = flux_y / (cgx*cgy*cgz);
        vmat(3, i) = flux_z / (cgx*cgy*cgz);
    end

    %% 4. Permeability Calculation
    % Map the recorded states to specific output variables
    state_x = states{1};
    state_y = states{2};
    state_z = states{3};

    % Compute the average pressure gradient matrix for the core
    % $\nabla P = \frac{\Delta P}{V_{core}}$
    core_vol = cgx * cgy * cgz;
    
    % Normalize the original gradient matrix by physical lengths
    for j = 1:d
        dp_mat(j, :) = dp_mat(j, :) / L(j);
    end
    
    % Replace diagonal with core-specific pressure drops
    dp_mat(1, 1) = dp_core(1) / core_vol;
    dp_mat(2, 2) = dp_core(2) / core_vol;
    dp_mat(3, 3) = dp_core(3) / core_vol;

    % Calculate Permeability: $K = V \cdot (\nabla P)^{-1}$
    perm = vmat / dp_mat;
end