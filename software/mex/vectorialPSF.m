%[h, dxp, dyp, dzp] = vectorialPSF(xp, z, nx, p) computes a vectorial microscope point spread function model.
% The partial derivatives of the model relative to the source position xp are also calculated.
% The model is described in [1]. For more information and implementation details, see [2].
%
%   INPUTS:
%   xp    : Source position, 3-element vector [xp yp zp]
%   z     : Vector of z-plane positions
%   nx    : Window size for the psf calculation, in pixels (must be odd).
%           The origin is located at ((nx+1)/2, (nx+1)/2).
%   p     : Parameter structure of system properties, with fields (case sensitive)
%            ti0       : working distance of the objective
%            ni0       : immersion medium refractive index, design value
%            ni        : immersion medium refractive index, experimental value
%            tg0       : coverslip thickness, design value
%            tg        : coverslip thickness, experimental value
%            ng0       : coverslip refractive index, design value
%            ng        : coverslip refractive index, experimental value
%            ns        : sample refractive index
%            lambda    : emission wavelength
%            M         : magnification
%            NA        : numerical aperture
%            pixelSize : physical size (width) of the camera pixels
%            f         : (optional, default: 3) oversampling factor to approximate pixel integration
%            mode      : (optional, default: 1) if 0, returns oversampled PSF
%
%   All spatial units are in object space, in [m].
%
%   Example structure for 'p':
%
%           p.ti0 = 1.9000e-04
%           p.ni0 = 1.5180
%            p.ni = 1.5180
%           p.tg0 = 1.7000e-04
%            p.tg = 1.7000e-04
%           p.ng0 = 1.5150
%            p.ng = 1.5150
%            p.ns = 1.46
%        p.lambda = 5.5000e-07
%             p.M = 100
%            p.NA = 1.4500
%     p.pixelSize = 6.4500e-06
%
% [1] F. Aguet et al., Opt. Express 17(8), pp. 6829-6848, 2009
% [2] F. Aguet, Ph.D Thesis, Swiss Federal Institute of Technology, Lausanne (EPFL), 2009
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

% Francois Aguet, 2009 (last modified: 07/31/2013)

function [h, dxp, dyp, dzp] = vectorialPSF(xp, z, nx, p) %#ok<STOUT,INUSD>