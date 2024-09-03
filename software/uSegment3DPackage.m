classdef uSegment3DPackage < Package
    % The main class of the uSegment3D Package
    %
    % Qiongjing (Jenny) Zou, June 2024
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
    
    methods
        function obj = uSegment3DPackage(owner,varargin)
            % Constructor of class uSegment3DPackage
            if nargin == 0
                super_args = {};
            else
                % Check input
                ip =inputParser;
                ip.addRequired('owner',@(x) isa(x,'MovieData'));
                ip.addOptional('outputDir',owner.outputDirectory_,@ischar);
                ip.parse(owner,varargin{:});
                outputDir = ip.Results.outputDir;
                
                super_args{1} = owner;
                super_args{2} = [outputDir  filesep 'uSegment3DPackage'];
            end
                 
            % Call the superclass constructor
            obj = obj@Package(super_args{:});        
        end
        
    end
    
    methods (Static)
        
        function name = getName()
            name = 'u-segment3D';
        end

        function m = getDependencyMatrix(i,j)
            %    1  2  3  4  5  {processes}
            m = [0  0  0  0  0  ;  % 1 ImagePreprocessingProcess
                 1  0  0  0  0  ;  % 2 SegmentationProcess    
                 0  1  0  0  0  ;  % 3 TwoDtoThreeDAggregationProcess
                 1  0  1  0  0  ;  % 4 SegmentationFilteringPostprocessingProcess
                 1  0  0  1  0  ]; % 5 SegmentationEnhancementPostprocessingProcess            
            if nargin<2, j=1:size(m,2); end
            if nargin<1, i=1:size(m,1); end
            m=m(i,j);
        end

        function varargout = GUI(varargin)
            % Start the package GUI
            varargout{1} = uSegment3DPackageGUI(varargin{:});
        end

        function procConstr = getDefaultProcessConstructors(index)
            procConstr = {
                @ImagePreprocessingProcess,...
                @CellposeSegmentationProcess,...
                @TwoDtoThreeDAggregationProcess,...
                @SegmentationFilteringPostprocessingProcess,...
                @SegmentationEnhancementPostprocessingProcess};
              
            if nargin == 0, index = 1 : numel(procConstr); end
            procConstr = procConstr(index);
        end

        function classes = getProcessClassNames(index)
            classes = {
                'ImagePreprocessingProcess',...
                'SegmentationProcess',...
                'TwoDtoThreeDAggregationProcess',...
                'SegmentationFilteringPostprocessingProcess',...
                'SegmentationEnhancementPostprocessingProcess'};        
            if nargin == 0, index = 1 : numel(classes); end
            classes = classes(index);
        end
    end  
end
