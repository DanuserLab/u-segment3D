classdef NonSingularProcess < Process
    %NONSINGULARPROCESS Process that passes itself as the first argument to
    %function and does not fall back to legacy behavior
    %
    % See also Process.run
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
    
    properties
    end
    
    methods
        function obj = NonSingularProcess(varargin)
            obj = obj@Process(varargin{:});
        end
        function runLegacy(obj,varargin)
            % Reset sucess flags and existing display methods
            % Runs the funName_ with MovieData handle as first argument
            obj.resetDisplayMethod();
            obj.success_=false;
            
            % Run the process in legacy mode with MovieData handle
            % as first argument!
            obj.startTime_ = clock;
            obj.funName_(obj.getOwner(), varargin{:});
            
            % Update flags and set finishTime
            obj.success_= true;
            obj.updated_= true;
            obj.procChanged_= false;
            obj.finishTime_ = clock;
            
            % Run sanityCheck on parent package to update dependencies
            for packId = obj.getPackageIndex()
                obj.getOwner().getPackage(packId).sanityCheck(false,'all');
            end
            
            obj.getOwner().save();
        end
    end
    
end

