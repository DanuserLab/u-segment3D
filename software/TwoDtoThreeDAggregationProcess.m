classdef  TwoDtoThreeDAggregationProcess < DataProcessingProcess & NonSingularProcess
    % Process Class for 2D to 3D Aggregation
    % twoDtoThreeDAggregateWrap.m is the wrapper function
    % TwoDtoThreeDAggregationProcess is part of uSegment3DPackage
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
        function obj = TwoDtoThreeDAggregationProcess(owner, varargin)
            
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
                super_args{2} = TwoDtoThreeDAggregationProcess.getName;
                super_args{3} = @twoDtoThreeDAggregateWrap;
                if isempty(funParams)
                    funParams = TwoDtoThreeDAggregationProcess.getDefaultParams(owner,outputDir);
                end
                super_args{4} = funParams;
            end
            obj = obj@DataProcessingProcess(super_args{:});
            obj.funName_ = super_args{3};
            obj.funParams_ = super_args{4};
            
            obj.is3Dcompatible_ = true;
        end
    end
    
    methods (Static)
        function name = getName()
            name = '2D to 3D Aggregation';
        end
        
        function h = GUI(varargin)
            h = @TwoDtoThreeDAggregationProcessGUI;
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
            funParams.OutputDirectory = [outputDir  filesep '2Dto3DAggregation'];
            funParams.ProcessIndex = []; % can use this parameter to set which previous process's output to be used as input for this process

            % Below parameters are created base on 
            % /python-applications/Segment3D/segment3D/parameters.py 
            funParams.gradient_descent.gradient_decay = 0.0;
            funParams.gradient_descent.do_mp = false;
            funParams.gradient_descent.tile_shape = {int16(128), int16(256), int16(256)}; % this parameter need to be a integer!; This is the equivalent to Python tuple class (128,256,256)
            funParams.gradient_descent.tile_aspect = {1,2,2}; % This is the equivalent to Python tuple class (1,2,2)
            funParams.gradient_descent.tile_overlap_ratio = 0.25;
            funParams.gradient_descent.n_iter = 200;
            funParams.gradient_descent.delta = 1.;
            funParams.gradient_descent.momenta = 0.98;
            funParams.gradient_descent.eps = 1e-12;
            funParams.gradient_descent.use_connectivity = false;
            funParams.gradient_descent.connectivity_alpha = 0.75;
            funParams.gradient_descent.interp = false;
            funParams.gradient_descent.binary_mask_gradient = false;
            funParams.gradient_descent.debug_viz = false;
            funParams.gradient_descent.renorm_gradient = false;
            funParams.gradient_descent.sampling = 1000;
            funParams.gradient_descent.track_percent = 0;
            funParams.gradient_descent.rand_seed = 0;
            funParams.gradient_descent.ref_initial_color_img = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.gradient_descent.ref_alpha = 0.5;
            funParams.gradient_descent.saveplotsfolder = string(missing);  % this is not empty, it will be set to outFilePaths{1,iChan} in wrapper fcn for each channel.
            funParams.gradient_descent.viewinit = {{0,0}}; % This is the equivalent to Python tuple class ((0,0))


            funParams.combine_cell_probs.ksize = int16(1); % this parameter need to be a integer!
            funParams.combine_cell_probs.alpha = 0.5;
            funParams.combine_cell_probs.eps = 1e-20;
            funParams.combine_cell_probs.cellpose_prob_mask = true;
            funParams.combine_cell_probs.smooth_sigma = 0;
            funParams.combine_cell_probs.threshold_level = int16(1); % this parameter need to be a integer!
            funParams.combine_cell_probs.threshold_n_levels = int16(3); % this parameter need to be a integer!
            funParams.combine_cell_probs.apply_one_d_p_thresh = true;
            funParams.combine_cell_probs.prob_thresh = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.combine_cell_probs.min_prob_thresh = 0.0;


            funParams.postprocess_binary.binary_closing = int16(1); % this parameter need to be a integer!
            funParams.postprocess_binary.remove_small_objects = int16(1000); % this parameter need to be a integer!
            funParams.postprocess_binary.binary_dilation = int16(1); % this parameter need to be a integer!
            funParams.postprocess_binary.binary_fill_holes = false;
            funParams.postprocess_binary.extra_erode = int16(0); % this parameter need to be a integer!


            funParams.combine_cell_gradients.ksize = int16(1); % this parameter need to be a integer!
            funParams.combine_cell_gradients.alpha = 0.5;
            funParams.combine_cell_gradients.eps = 1e-20;
            funParams.combine_cell_gradients.smooth_sigma = 1;
            funParams.combine_cell_gradients.post_sigma = 0;


            funParams.connected_component.min_area = int16(5); % this parameter need to be a integer!
            funParams.connected_component.smooth_sigma = 1.;
            funParams.connected_component.thresh_factor = 0;


            funParams.indirect_method.dtform_method = 'cellpose_improve';
            funParams.indirect_method.iter_factor = int16(5); % this parameter need to be a integer!
            funParams.indirect_method.power_dist = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.indirect_method.n_cpu = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.indirect_method.edt_fixed_point_percentile = 0.01;
            funParams.indirect_method.smooth_binary = 1;
            funParams.indirect_method.smooth_skel_sigma = 3;

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
