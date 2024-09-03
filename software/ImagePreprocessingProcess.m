classdef  ImagePreprocessingProcess < ImageProcessingProcess & NonSingularProcess
    % Process Class for Image Preprocessing
    % imagePreprocessWrap.m is the wrapper function
    % ImagePreprocessingProcess is part of uSegment3DPackage
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
        function obj = ImagePreprocessingProcess(owner, varargin)
            
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
                super_args{2} = ImagePreprocessingProcess.getName;
                super_args{3} = @imagePreprocessWrap;
                if isempty(funParams)
                    funParams = ImagePreprocessingProcess.getDefaultParams(owner,outputDir);
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
            name = 'Image Preprocessing';
        end
        
        function h = GUI(varargin)
            h = @ImagePreprocessingProcessGUI;
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
            funParams.OutputDirectory = [outputDir  filesep 'ImagePreprocessing'];
            
            % Below parameters are created base on 
            % /python-applications/Segment3D/segment3D/parameters.py 
            funParams.factor = 1;
            funParams.voxel_res = [1,1,1];
            funParams.do_bg_correction = true;
            funParams.bg_ds = 16;
            funParams.bg_sigma = 5;
            funParams.normalize_min = 2.;
            funParams.normalize_max = 99.8;
            funParams.do_avg_imgs = false;
            funParams.avg_func_imgs = 'np.nanmean'; % Put this as string now and convert to numpy fcn in the main python script.

        end

    end
end
