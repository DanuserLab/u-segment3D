classdef  CellposeSegmentationProcess < SegmentationProcess & NonSingularProcess
    % Process Class for Cellpose Segmentation
    % cellposeSegmentWrap.m is the wrapper function
    % CellposeSegmentationProcess is part of uSegment3DPackage
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
        function obj = CellposeSegmentationProcess(owner, varargin)
            
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
                super_args{2} = CellposeSegmentationProcess.getName;
                super_args{3} = @cellposeSegmentWrap;
                if isempty(funParams)
                    funParams = CellposeSegmentationProcess.getDefaultParams(owner,outputDir);
                end
                super_args{4} = funParams;
            end
            obj = obj@SegmentationProcess(super_args{:});
            obj.funName_ = super_args{3};
            obj.funParams_ = super_args{4};
            
            obj.is3Dcompatible_ = true;
        end
    end
    
    methods (Static)
        function name = getName()
            name = 'Cellpose Segmentation';
        end
        
        function h = GUI(varargin)
            h = @CellposeSegmentationProcessGUI;
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
            funParams.OutputDirectory = [outputDir  filesep 'CellposeSegmentation'];
            funParams.ProcessIndex = []; % can use this parameter to set which previous process's output to be used as input for this process

            % Below parameters are created base on 
            % /python-applications/Segment3D/segment3D/parameters.py            
            funParams.hist_norm = false;
            funParams.cellpose_modelname = 'cyto';
            funParams.cellpose_channels = 'grayscale';
            funParams.ksize = int16(15); % this parameter need to be a integer!
            funParams.use_Cellpose_auto_diameter = false;
            funParams.gpu = true;
            funParams.best_diam = string(missing); % This is the equivalent to Python None in MATLAB.
            funParams.model_invert = false;
            funParams.test_slice = string(missing);
            funParams.diam_range = 'np.arange(10,121,2.5)'; % Put this as string now and convert to numpy fcn in the main python script.
            funParams.smoothwinsize = int16(5); % this parameter need to be a integer!
            funParams.histnorm_kernel_size = {int16(64), int16(64)}; % this parameter need to be a integer!; This is the equivalent to Python tuple class (64, 64)
            funParams.histnorm_clip_limit = 0.05;
            funParams.use_edge = true;
            funParams.show_img_bounds = {int16(1024), int16(1024)}; % this parameter need to be a integer!; This is the equivalent to Python tuple class (1024, 1024)
            funParams.saveplotsfolder = string(missing); % this is not empty, it will be set to outFilePaths{1,iChan} in wrapper fcn for each channel.
            funParams.use_prob_weighted_score = true;
            funParams.debug_viz = true;

        end

        function validTypes =  getValidCellposeModelname()
            validTypes = {'cyto', ...    
                          'cyto2',... 
                          'cyto3',...
                          'nuclei'};
        end

        function validTypes =  getValidCellposeChannels()
            validTypes = {'grayscale', ...    
                          'color'};
        end

        function validTypes =  getValidUseEdge()
            validTypes = {'Use edge strength to determine optimal slice', ...    
                          'Use maximum intensity to determine optimal slice'};
        end

    end
end
