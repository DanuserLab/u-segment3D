classdef  SegmentationEnhancementPostprocessingProcess < ImageProcessingProcess & NonSingularProcess
    % Process Class for Segmentation Enhancement Postprocessing
    % segmentEnhancementPostprocessWrap.m is the wrapper function
    % SegmentationEnhancementPostprocessingProcess is part of uSegment3DPackage
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
        function obj = SegmentationEnhancementPostprocessingProcess(owner, varargin)
            
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
                super_args{2} = SegmentationEnhancementPostprocessingProcess.getName;
                super_args{3} = @segmentEnhancementPostprocessWrap;
                if isempty(funParams)
                    funParams = SegmentationEnhancementPostprocessingProcess.getDefaultParams(owner,outputDir);
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
            name = 'Segmentation Enhancement Postprocessing';
        end
        
        function h = GUI(varargin)
            h = @SegmentationEnhancementPostprocessingProcessGUI;
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
            funParams.OutputDirectory = [outputDir  filesep 'SegmentationEnhancementPostprocessing'];
            funParams.ProcessIndex = []; % can use this parameter to set which previous process's output to be used as input for this process

            % Below parameters are created base on 
            % /python-applications/Segment3D/segment3D/parameters.py 

            % label_diffusion_params:
            funParams.diffusion.n_cpu = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.diffusion.refine_clamp = 0.7;
            funParams.diffusion.refine_iters = int16(50); % this parameter need to be a integer!
            funParams.diffusion.refine_alpha = 0.5;
            funParams.diffusion.pad_size = int16(25); % this parameter need to be a integer!
            funParams.diffusion.multilabel_refine = false;
            funParams.diffusion.noprogress_bool = true;
            funParams.diffusion.affinity_type = 'heat';

            funParams.guide_img.pmin = 0;
            funParams.guide_img.pmax = 100;


            % guided_filter_params:
            funParams.guide_filter.radius = int16(25); % this parameter need to be a integer!
            funParams.guide_filter.eps = 1e-4;
            funParams.guide_filter.n_cpu = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.guide_filter.pad_size = int16(25); % this parameter need to be a integer!
            funParams.guide_filter.size_factor = 0.75;
            funParams.guide_filter.min_protrusion_size = 15;
            funParams.guide_filter.adaptive_radius_bool = false;
            funParams.guide_filter.mode = 'normal';
            funParams.guide_filter.base_dilate = int16(0); % this parameter need to be a integer!
            funParams.guide_filter.base_erode = int16(5); % this parameter need to be a integer!
            funParams.guide_filter.collision_erode = int16(2); % this parameter need to be a integer!
            funParams.guide_filter.collision_close = int16(3); % this parameter need to be a integer!
            funParams.guide_filter.collision_dilate = int16(0); % this parameter need to be a integer!
            funParams.guide_filter.collision_fill_holes = true;
            funParams.guide_filter.threshold_level = int16(0); % this parameter need to be a integer!
            funParams.guide_filter.threshold_nlevels = int16(2); % this parameter need to be a integer!
            funParams.guide_filter.use_int = false;

            funParams.ridge_filter.sigmas = {3}; % This is the equivalent to Python list [3]
            funParams.ridge_filter.black_ridges = false;
            funParams.ridge_filter.mix_ratio = 0.5;
            funParams.ridge_filter.do_ridge_enhance = false;
            funParams.ridge_filter.do_multiprocess_2D = true;
            funParams.ridge_filter.low_contrast_fraction = 0.05;
            funParams.ridge_filter.n_cpu = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.ridge_filter.pmin = 2;
            funParams.ridge_filter.pmax = 99.8;


            funParams.guide_img2.pmin = 0;
            funParams.guide_img2.pmax = 100;

        end

        function validTypes =  getValidGuideFilterModename()
            validTypes = {'normal', ...    
                          'additive'};
        end

    end
end
