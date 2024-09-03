classdef  SegmentationFilteringPostprocessingProcess < ImageProcessingProcess & NonSingularProcess
    % Process Class for Segmentation Filtering Postprocessing
    % segmentFilteringPostprocessWrap.m is the wrapper function
    % SegmentationFilteringPostprocessingProcess is part of uSegment3DPackage
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
    
    methods (Access = public)
        function obj = SegmentationFilteringPostprocessingProcess(owner, varargin)
            
            if nargin == 0
                super_args = {};
            else
                % Input check
                ip = inputParser;
                ip.CaseSensitive = false;
                ip.KeepUnmatched = true;
                ip.addRequired('owner',@(x) isa(x,'MovieData'));
                ip.addOptional('outputDir',owner.outputDirectory_,@ischar);
                ip.addOptional('funParams',[],@isstruct);
                ip.parse(owner,varargin{:});
                outputDir = ip.Results.outputDir;
                funParams = ip.Results.funParams;
                
                super_args{1} = owner;
                super_args{2} = SegmentationFilteringPostprocessingProcess.getName;
                super_args{3} = @segmentFilteringPostprocessWrap;
                if isempty(funParams)
                    funParams = SegmentationFilteringPostprocessingProcess.getDefaultParams(owner,outputDir);
                end
                super_args{4} = funParams;
            end
            obj = obj@ImageProcessingProcess(super_args{:});
            obj.funName_ = super_args{3};
            obj.funParams_ = super_args{4};
            
            obj.is3Dcompatible_ = true;
        end
    end
    
    methods (Static)
        function name = getName()
            name = 'Segmentation Filtering Postprocessing';
        end
        
        function h = GUI(varargin)
            h = @SegmentationFilteringPostprocessingProcessGUI;
        end
        
        function funParams = getDefaultParams(owner, varargin)
            % Input check
            ip=inputParser;
            ip.addRequired('owner',@(x) isa(x,'MovieData'));
            ip.addOptional('outputDir', owner.outputDirectory_, @ischar);
            ip.parse(owner, varargin{:})
            outputDir = ip.Results.outputDir;
            
            % Set default parameters
            funParams.ChannelIndex = 1:numel(owner.channels_);
            funParams.OutputDirectory = [outputDir  filesep 'SegmentationFilteringPostprocessing'];
            funParams.ProcessIndex = []; % can use this parameter to set which previous process's output to be used as input for this process

            % Below parameters are created base on 
            % /python-applications/Segment3D/segment3D/parameters.py 
            funParams.size_filters.min_size = 200;
            funParams.size_filters.max_size_factor = 10;
            funParams.size_filters.do_stats_filter = true;


            funParams.flow_consistency.flow_threshold = 0.85;
            funParams.flow_consistency.do_flow_remove = true;
            funParams.flow_consistency.edt_fixed_point_percentile = 0.01;
            funParams.flow_consistency.dtform_method = 'cellpose_improve';
            funParams.flow_consistency.iter_factor = int16(5); % this parameter need to be a integer!
            funParams.flow_consistency.power_dist = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.flow_consistency.smooth_skel_sigma = 3;
            funParams.flow_consistency.n_cpu = string(missing); % This is the equivalent to Python None in MATLAB.


        end
        
        function validTypes =  getValidDistTransMethod()
            validTypes = {'cellpose_improve', ...   % Heat Diffusion
                          'edt',...                 % Euclidean Distance Transform
                          'fmm',...                 % Geodesic Centroid
                          'fmm_skel',...            % Geodesic Skeleton
                          'cellpose_skel'};         % Diffusion Skeleton
        end

    end
end
