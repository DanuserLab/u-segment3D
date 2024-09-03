function cluster =  uTrackParCluster(cluster)
%uTrackParCluster Set or Query the parallel.Cluster that uTrack should use
%for launching jobs via batch or createJob.
%
% Parallel pools may still use the default (local?) cluster set by matlab
%
% See also parcluster
%
% Mark Kittisopikul, January 2016
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
    persistent currentCluster;
    if(nargin < 1)
%         if(isempty(currentCluster))
%             % if cluster has not been set, use the default cluster
%             currentCluster = parcluster;
%         end
    else
        if(ischar(cluster))
            % attempt to convert a string profile name to a parallel
            % cluster
            cluster = parcluster(cluster);
        end
        assert(isempty(cluster) || isa(cluster,'parallel.Cluster'), ...
            'uTrackParCluster:Argument must be a parallel.Cluster, a char, or empty');
        currentCluster = cluster;
    end
    % return the current cluster stored here
    cluster = currentCluster;
end