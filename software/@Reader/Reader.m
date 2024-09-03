classdef  Reader < handle & matlab.mixin.Copyable
    % Reader is an abstract class for reading image data in a plane-wise
    % fashion
    %
    % Arguments are generally in the order C, T, Z
    %
    % Properties
    % ----------
    % (use the get methods rather than these properties)
    % sizeX, sizeY, sizeZ, sizeC, sizeT, bitDepth
    %
    % Reader methods:
    %
    % Public Methods
    % --------------
    % loadImage(c, t, [z], ...) - loads an image plane at channel c, time
    % point t, and optionally at z-plane z (default: 1)
    % 
    % loadStack(c, t, [Z], ...) -  loads a stack of images in channel c and
    % timepoint t. Z is optional and is an array of integers indicating a
    % subset of Z planes (default: 1:getSizeZ).
    %
    % Public Abstract Methods
    % -----------------------
    % (see subclasses, generally require no arguments unless indicated)
    % getSizeX - length of the x, horizontal,  dimension as an integer
    % getSizeY - length of the y, vertical, dimension as an integer
    % getSizeZ - length of the z dimension as an integer
    % getSizeC - the number of channels as an integer
    % getSizeT - the number of time points as an integer
    % getBitDepth - the bitdepth as an integer
    % getImageFileNames(c,t) - the filenames of the underlying images as a cell
    % array of character strings
    % getChannelNames(c) - the names of the channels from metadata as a cell
    % array of character strings; c is an array of the channels of interest
    % 
    % Known subclasses
    % ----------------
    %
    % BioFormatsReader
    % TiffSeriesReader
    % HCSReader
    % ProxyReader and subclasses (CellReader, etc.)
    % 
    % See also BioFormatsReader, TiffSeriesReader, HCSReader, ProxyReader,
    % CellReader
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
    
    % Notes for Implementing Subclasses
    %
    % Subclasses should consider consider overloading
    % protected methods ending with an underscore, '_'
    % such as loadImage_ and loadStack_.
    %
    % The public loadImage and loadStack methods contain
    % input validation and default parameters which should
    % be changed minimally.
    
    properties
        sizeX
        sizeY
        sizeZ
        sizeC
        sizeT
        bitDepth
    end
       
    methods(Abstract)
        getSizeX(obj)
        getSizeY(obj)
        getSizeZ(obj)
        getSizeC(obj)
        getSizeT(obj)
        getBitDepth(obj)
        getImageFileNames(obj)
        getChannelNames(obj)
    end
    
    methods ( Access = public )
        function B = sizeCheckNeed(obj)
            B=true;
        end
        
        function I = loadImage(obj, c, t, varargin)
        % loadImage reads a single image plane as a 2D, YX Matrix
        %
        % loadImage(c,t,z) reads the YX plane at (c,t,z)
        %
        % loadImage(c,t) reads the YX plane at (c,t,1)
        %
        % Note: Override if you need to overload or change how the input is
        % checked. Otherwise override loadImage_.
        %
        % Example:
        %   reader = movieData.getReader();
        %   I = reader.loadImage(1,1);
        %   imshow(I,[]);
        %
        
        % Backwards compatability before 2015/01/01:
        % Previously this function was abstract and therefore should be
        % overridden by all subclasses written prior to 2015/01/01
         
            ip = inputParser;
            if(~obj.sizeCheckNeed())
                ip.addRequired('c', @isnumeric);
                ip.addRequired('t', @isnumeric);
                ip.addRequired('z', @isnumeric);
            else
            ip.addRequired('c', ...
                @(c) insequence_and_scalar(c, 1,   obj.getSizeC() ) );
            ip.addRequired('t', ... 
                @(t) insequence_and_scalar(t, 1, obj.getSizeT()));
            ip.addOptional('z', 1, ...
                @(z) insequence_and_scalar(z, 1,  obj.getSizeZ()) || ...
                    isempty(z));
            end
            ip.parse(c, t, varargin{:});
            
            z = ip.Results.z;
            if(isempty(z))
                z = 1;
            end
            
            % Flag to avoid infinite loops
            readerLoadImageFlag = true;
            
            I = obj.loadImage_(c , t , z);
        end
        
        function I = loadStack(obj, c, t, varargin)
        % loadStack reads a Z-stack and returns a YXZ Matrix
        %
        %   loadStack(c,t,Z) returns a Z-stack for channel c and
        %   time frame t  with Z planes indicated by the vector Z
        %
        %   loadStack(c,t) returns loadStack(c,t, 1: getSizeZ())
        %
        % output: a 3D matrix with dimensions YXZ
        %
        % Note: Override only if need you need to overload or change how
        % the input is checked. Otherwise override loadStack_.
        %
        % Example:
        %   reader = movieData.getReader();
        %   % The following are all equivalent
        %   zStackMatrix = reader.loadStack(1,1,1:reader.getSizeZ());
        %   zStackMatrix = reader.loadStack(1,1);
        %   zStackMatrix = reader.loadStack(1);
        %
        
            % Input check
            ip = inputParser;
            if(any(cellfun(@(s) strcmp(s,'ZSize'),(varargin(cellfun(@ischar,varargin))))))
                ip.addRequired('c', @isnumeric);
                ip.addRequired('t', @isnumeric);
                ip.addOptional('z',[], @isnumeric);
                ip.addParamValue('ZSize',[], @isnumeric);
                ip.parse(c, t, varargin{:});
                ZSize=ip.Results.ZSize;
            else
                ZSize=obj.getSizeZ();
                ip.addRequired('c', ...
                    @(x) insequence_and_scalar(x, 1, obj.getSizeC() ) );
                ip.addRequired('t', ...
                    @(x) insequence_and_scalar(x, 1, obj.getSizeT() ) );
                ip.addOptional('z', [] , ...
                    @(x) all_insequence(x,1,      ZSize )  );
                ip.parse(c, t, varargin{:});
            end

            z = ip.Results.z;
            
            if(isempty(z))
                z = 1 : ZSize;
            end
            
            I = obj.loadStack_(c,t,z);
        end
    end
    methods( Access = protected )
        function I = loadStack_(obj, c, t, z)
        % loadStack_ is a validation-free version called by loadStack
        %
        % Provides a generic implementation of loadStack.
        % Guarantees all readers will have a loadStack method if they have
        % a loadImage method.
        %
        % Override if there are backend optimizations.
        % This is likely slower because of repeated input validation.
        %
        % parameters are c, t, z with only c required
        % t defaults to 1
        % z defaults to 1:sizeZ
        %
        
            % Load first image to get class and dimensions.
            first = obj.loadImage_( c , t , z(1) );
            % "I" will be a YXZ matrix.
            I = zeros( [ size(first) length(z) ] , class(first));
            I(:,:,1) = first;
            
            for zi = 2:length(z)
                I(:,:,zi) = obj.loadImage_( c , t , z(zi) );
            end
                
        end
    %end
    %methods( Abstract, Access = protected )
        % I = loadImage_(obj, c, t, z )
        function I = loadImage_(obj, c, t, z )
            % loadImage_ is a validation-free implementation called by
            %   loadImage
            %
            % Forward to validation version.
            % Either loadImage_ (recommended) or loadImage must be
            %   overloaded by subclasses
            
            % Avoid infinite loop
            if(evalin('caller','readerLoadImageFlag'))
                error('Reader::loadImage_', ...
                    ['Reader subclasses must either overload' ... 
                    'loadImage or loadImage_']);
            else
                I = obj.loadImage(c,t,z);
            end
        end
    end
    
end
