%% Steady-state Permeability Upscaling: Extended Local Method
% This script demonstrates single-phase permeability upscaling using an 
% extended local domain (buffer/ring) to reduce boundary effects.

mrstModule add mimetic upscaling spe10 incomp deckformat

%% 1. Grid Setup with Extended Local Buffer
% cg: Dimensions of the target core block
% ringSize: Buffer layers added to each side [x, y, z]
cg       = [5, 5, 2];
ringSize = [4, 4, 0]; 
elcg     = cg + ringSize;

G = cartGrid(elcg);
G = computeGeometry(G);

% Define periodic boundary pairs (Left-Right, Front-Back, Top-Bottom)
bcr = cell(3,1); bcl = cell(3,1);
bcr{1} = pside([], G, 'RIGHT',  0); bcl{1} = pside([], G, 'LEFT',   0);
bcr{2} = pside([], G, 'FRONT',  0); bcl{2} = pside([], G, 'BACK',   0);
bcr{3} = pside([], G, 'BOTTOM', 0); bcl{3} = pside([], G, 'TOP',    0);

% Construct the periodic grid object
[Gp, bcp] = makePeriodicGridMulti3d(G, bcl, bcr, {0, 0, 0});

%% 2. Load SPE10 Rock Properties
% Extract a subset of the SPE10 model starting at (x,y,z)
x0 = 15; y0 = 15; z0 = 20;
rock = getSPE10rock(x0 : (x0 - 1 + G.cartDims(1)), ...
                    y0 : (y0 - 1 + G.cartDims(2)), ...
                    z0 : (z0 - 1 + G.cartDims(3)));

rock.perm        = convertTo(rock.perm, milli*darcy);
rock.poro(1:end) = 0.2;

% Visualize the permeability of the extended domain
clf
plotCellData(G, log10(rock.perm(:,1))); 
view(3); axis tight; colorbar;
title('Fine-scale Permeability (Extended Domain)');

%% 3. Single-Phase Upscaling Calculation
% Setup a unit fluid (viscosity = 1, density = 1)
fluid_pure = initSingleFluid('mu', 1*cP, 'rho', 1000*kg/m^3);

% Define the Pressure Solver (TPFA)
% Note: computeTransGp handles transmissibility for periodic grids
T_periodic = computeTransGp(G, Gp, rock);
psolver    = @(state, Grid, Fluid, BCP, Rock) ...
              incompTPFA(state, Grid, T_periodic, Fluid, 'bcp', BCP);

% Physical dimensions of the extended grid
L = max(G.faces.centroids) - min(G.faces.centroids);

% Execute Upscaling
fprintf('Starting Extended Local Upscaling...\n');
warning('off', 'mrst:periodic_bc');
tic;

permEL = upscalePermeabilityPeriodicEL(...
    Gp, bcp, 1*barsa, psolver, fluid_pure, rock, L, ringSize);

toc;
warning('on', 'mrst:periodic_bc');

%% 4. Results Display
fprintf('\nUpscaled Permeability Tensor (mD):\n');
disp(permEL / (milli*darcy));

% Optional: Plot flux magnitude from the last solve
clf
plotFaceData(G, abs(stateEL.flux)); 
view(3); axis tight; colorbar;
title('Flux Magnitude (Final Solve)');