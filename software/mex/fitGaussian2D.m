%FITGAUSSIAN2D Fit a 2-D Gaussian function to data in a square image window.
%    [prmVect prmStd C res J] = fitGaussian2D(data, prmVect, mode, options)
%
%    Symbols: xp : x-position
%             yp : y-position
%              A : amplitude
%              s : standard deviation
%              c : background
%
%    The origin is defined at the center of the input window.
%
%    Inputs:     data : 2-D image array
%             prmVect : parameter vector with order: [xp, yp, A, s, c]
%                mode : string that defines parameters to be optimized; any among 'xyasc'
%           {options} : vector [maxIter eAbs eRel]; max. iterations, tolerances. See GSL documentation.
%
%    Pixels in 'data' that are set to NaN are masked in the optimization.
%
%    Outputs: prmVect : parameter vector
%              prmStd : parameter standard deviations
%                   C : covariance matrix
%                 res : structure with fields:
%                         .data : residuals
%                         .pval : p-value of the Kolmogorov-Smirnov test (normal dist.)
%                         .mean : mean of the residuals
%                         .std  : standard deviation of the residuals
%                         .RSS  : residual sum of squares
%                   J : Jacobian
%
% Axis conventions: image processing, see meshgrid
% For Gaussian mixture fitting, use fitGaussianMixture2D()
%
% Example: [prmVect prmStd C res J] = fitGaussian2D(data, [0 0 max(data(:)) 1.5 min(data(:))], 'xyasc');
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

% Francois Aguet, 2011 (last modified May 1, 2011)