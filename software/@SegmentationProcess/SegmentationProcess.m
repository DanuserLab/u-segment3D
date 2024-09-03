classdef SegmentationProcess < MaskProcess
    % An abstract superclass of all segmentation processes
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

    % Sebastien Besson 4/2011
    
    methods (Access = protected)
        function obj = SegmentationProcess(owner,name,funName, funParams,...
                outFilePaths)
            % Constructor of class SegmentationProcess
            if nargin == 0
                super_args = {};
            else
                super_args{1} = owner;
                super_args{2} = name;
            end
            if nargin > 2
                super_args{3} = funName;
            end
            if nargin > 3
                super_args{4} = funParams;
            end
            if nargin > 5
                super_args{5} = outFilePaths;
            end
            % Call the superclass constructor - these values are private
            obj = obj@MaskProcess(super_args{:});
           
        end
    end
    methods(Static)
        function name =getName()
            name = 'Segmentation';
        end
        function h = GUI()
            h= @abstractProcessGUI;
        end
        function procClasses = getConcreteClasses(varargin)
            procClasses = ...
                {@ThresholdProcess;
                 @MultiScaleAutoSegmentationProcess;
                 @ExternalSegmentationProcess;
                 @CellposeSegmentationProcess; % uSegment3DPackage only
                 @ExternalSegment3DProcess; % uSegment3DPackage only
                 @ThresholdProcess3D;
                };

           % If input, check if 2D or 3D movie(s).
            ip =inputParser;
            ip.addOptional('MO', [], @(x) isa(x,'MovieData') || isa(x,'MovieList'));
            ip.parse(varargin{:});
            MO = ip.Results.MO;

            if ~isempty(MO)
                if isa(MO,'MovieList')
                    MD = MO.getMovie(1);
                elseif length(MO) > 1
                    MD = MO(1);
                else
                    MD = MO;
                end

                if isempty(MD)
                    warning('MovieData properties not specified (2D vs. 3D)');
                    disp('Displaying both 2D and 3D Segmentation processes');
                else
                    if ~isempty(MD.getPackageIndex('uSegment3DPackage'))
                        disp('Displaying Segmentation processes for uSegment3DPackage only');
                        procClasses([1:3, 6]) = [];
                    elseif MD.is3D
                        disp('Detected 3D movie');
                        disp('Displaying 3D Segmentation processes only');
                        procClasses([1:2, 4:5]) = [];                        
                    elseif ~MD.is3D
                        disp('Detected 2D movie');
                        disp('Displaying 2D Segmentation processes only');
                        procClasses(4:6) = [];
                    end
                end

            end
            procClasses = cellfun(@func2str, procClasses, 'Unif', 0);
        end
        
    end
end
