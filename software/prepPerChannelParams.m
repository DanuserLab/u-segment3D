function params = prepPerChannelParams(params,nChan)
%PREPPERCHANNELPARAMS sets up per-channel parameter structure given variable input format
%
% params = prepPerChannelParams(params,nChan)
%
% This is for functions which have parameters which may vary for each
% channel in a dataset, or which may also be constant across channels. 
%
% It expects the input structure to have a field "PerChannelParams" which
% is a cell array of field names, and replictes them if necessary so that
% all fields with these names have exactly nChan columns (or elements if a
% cell array).
%
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

%Hunter Elliott
%2013

if isfield(params,'PerChannelParams') && ~isempty(params.PerChannelParams) && iscell(params.PerChannelParams);
      
    nPCPar = numel(params.PerChannelParams);
    
    for j = 1:nPCPar        
        if isfield(params,params.PerChannelParams{j})
            %nEl = numel(params.(params.PerChannelParams{j}));
            if ischar(params.(params.PerChannelParams{j})) 
                params.(params.PerChannelParams{j}) = {params.(params.PerChannelParams{j})};
            end
            nEl = size(params.(params.PerChannelParams{j}),2);
            if  nEl == 1
                params.(params.PerChannelParams{j}) = repmat(params.(params.PerChannelParams{j}),[1 nChan]);
            elseif nEl ~= nChan
                try
                    warning(['The parameter "' params.PerChannelParams{j} '" was designated as a per-channel parameter, but contained ' num2str(nEl) ' elements - this must be specified as either a scalar or have array of size equal to the number of channels!'])
                    params.(params.PerChannelParams{j}) = repmat({params.(params.PerChannelParams{j})},[1 nChan]);
                catch
                    error(['The parameter "' params.PerChannelParams{j} '" was designated as a per-channel parameter, but contained ' num2str(nEl) ' elements - this must be specified as either a scalar or have array of size equal to the number of channels!'])    
                end
            end                                    
        end
    end        
end
