function [outlierIdx,inlierIdx] = detectOutliers(observations,k)
%DETECTOUTLIERS detects outliers in a dataset
%
%SYNPOSIS [outlierIdx,inlierIdx] = detectOutliers(observations,k)
%
%INPUT  observations: Vector of observations (dataset).
%       k           : Roughly, for a certain value of k, observations
%                     that are k*sigma away from the mean will be
%                     considered outliers. 
%                     Optional. Default: 3.
%OUTPUT outlierIdx  : Index of observations that are considered outliers.
%       inlierIdx   : Index of observations that are considered inliers.
%
%REMARKS See Danuser 1992 or Rousseeuw & Leroy 1987 for details of
%        algorithm.
%
%Khuloud Jaqaman, October 2007
%Hunter Elliott, 6/4/2009  - switched median to nanmedian to support
%missing observations.
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

if nargin < 2 || isempty(k)
    k = 3;
end

%% outlier detection

%calculate median of observations
medObs = nanmedian(observations(:));

%get residuals, i.e. distance of observations from median
residuals = observations(:) - medObs;

%square the residuals
res2 = residuals .^ 2;

%calculate the median of the squared residuals
medRes2 = max(nanmedian(res2),eps);

%define parameter to remove outliers (see Rousseeuw 1987, p. 202)
magicNumber2 = 1/norminv(.75)^2;

%calculate test-statistic values
testValue = res2 / (magicNumber2 * medRes2);

%determine which observations are inliers and which are outliers
inlierIdx = find(testValue <= k^2);
outlierIdx = find(testValue > k^2);

%Make sure these damn things are row vectors - somehow it returns them
%differently with different input
inlierIdx = inlierIdx(:)';
outlierIdx = outlierIdx(:)';

%% ~~~ the end ~~~