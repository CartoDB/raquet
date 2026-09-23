/**
 * JPEG Decoder for RaQuet - Pure JavaScript JPEG decoding
 * Based on jpeg-js decoder (Apache 2.0 License)
 * Copyright 2011 notmasteryet
 *
 * Modified for BigQuery/Snowflake UDF environments (no Buffer dependency)
 */

var jpegDecoder = (function() {
  "use strict";

  var dctZigZag = new Int32Array([
     0,
     1,  8,
    16,  9,  2,
     3, 10, 17, 24,
    32, 25, 18, 11, 4,
     5, 12, 19, 26, 33, 40,
    48, 41, 34, 27, 20, 13,  6,
     7, 14, 21, 28, 35, 42, 49, 56,
    57, 50, 43, 36, 29, 22, 15,
    23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31,
    39, 46, 53, 60,
    61, 54, 47,
    55, 62,
    63
  ]);

  var dctCos1  =  4017;
  var dctSin1  =   799;
  var dctCos3  =  3406;
  var dctSin3  =  2276;
  var dctCos6  =  1567;
  var dctSin6  =  3784;
  var dctSqrt2 =  5793;
  var dctSqrt1d2 = 2896;

  var maxMemoryUsageBytes = 0;
  var totalBytesAllocated = 0;

  function requestMemoryAllocation(increaseAmount) {
    increaseAmount = increaseAmount || 0;
    var totalMemoryImpactBytes = totalBytesAllocated + increaseAmount;
    if (totalMemoryImpactBytes > maxMemoryUsageBytes) {
      var exceededAmount = Math.ceil((totalMemoryImpactBytes - maxMemoryUsageBytes) / 1024 / 1024);
      throw new Error('maxMemoryUsageInMB limit exceeded by at least ' + exceededAmount + 'MB');
    }
    totalBytesAllocated = totalMemoryImpactBytes;
  }

  function buildHuffmanTable(codeLengths, values) {
    var k = 0, code = [], i, j, length = 16;
    while (length > 0 && !codeLengths[length - 1])
      length--;
    code.push({children: [], index: 0});
    var p = code[0], q;
    for (i = 0; i < length; i++) {
      for (j = 0; j < codeLengths[i]; j++) {
        p = code.pop();
        p.children[p.index] = values[k];
        while (p.index > 0) {
          if (code.length === 0)
            throw new Error('Could not recreate Huffman Table');
          p = code.pop();
        }
        p.index++;
        code.push(p);
        while (code.length <= i) {
          code.push(q = {children: [], index: 0});
          p.children[p.index] = q.children;
          p = q;
        }
        k++;
      }
      if (i + 1 < length) {
        code.push(q = {children: [], index: 0});
        p.children[p.index] = q.children;
        p = q;
      }
    }
    return code[0].children;
  }

  function decodeScan(data, offset, frame, components, resetInterval,
                      spectralStart, spectralEnd, successivePrev, successive, opts) {
    var mcusPerLine = frame.mcusPerLine;
    var progressive = frame.progressive;
    var startOffset = offset, bitsData = 0, bitsCount = 0;

    function readBit() {
      if (bitsCount > 0) {
        bitsCount--;
        return (bitsData >> bitsCount) & 1;
      }
      bitsData = data[offset++];
      if (bitsData == 0xFF) {
        var nextByte = data[offset++];
        if (nextByte) {
          throw new Error("unexpected marker: " + ((bitsData << 8) | nextByte).toString(16));
        }
      }
      bitsCount = 7;
      return bitsData >>> 7;
    }

    function decodeHuffman(tree) {
      var node = tree, bit;
      while ((bit = readBit()) !== null) {
        node = node[bit];
        if (typeof node === 'number')
          return node;
        if (typeof node !== 'object')
          throw new Error("invalid huffman sequence");
      }
      return null;
    }

    function receive(length) {
      var n = 0;
      while (length > 0) {
        var bit = readBit();
        if (bit === null) return;
        n = (n << 1) | bit;
        length--;
      }
      return n;
    }

    function receiveAndExtend(length) {
      var n = receive(length);
      if (n >= 1 << (length - 1))
        return n;
      return n + (-1 << length) + 1;
    }

    function decodeBaseline(component, zz) {
      var t = decodeHuffman(component.huffmanTableDC);
      var diff = t === 0 ? 0 : receiveAndExtend(t);
      zz[0]= (component.pred += diff);
      var k = 1;
      while (k < 64) {
        var rs = decodeHuffman(component.huffmanTableAC);
        var s = rs & 15, r = rs >> 4;
        if (s === 0) {
          if (r < 15)
            break;
          k += 16;
          continue;
        }
        k += r;
        var z = dctZigZag[k];
        zz[z] = receiveAndExtend(s);
        k++;
      }
    }

    function decodeDCFirst(component, zz) {
      var t = decodeHuffman(component.huffmanTableDC);
      var diff = t === 0 ? 0 : (receiveAndExtend(t) << successive);
      zz[0] = (component.pred += diff);
    }

    function decodeDCSuccessive(component, zz) {
      zz[0] |= readBit() << successive;
    }

    var eobrun = 0;

    function decodeACFirst(component, zz) {
      if (eobrun > 0) {
        eobrun--;
        return;
      }
      var k = spectralStart, e = spectralEnd;
      while (k <= e) {
        var rs = decodeHuffman(component.huffmanTableAC);
        var s = rs & 15, r = rs >> 4;
        if (s === 0) {
          if (r < 15) {
            eobrun = receive(r) + (1 << r) - 1;
            break;
          }
          k += 16;
          continue;
        }
        k += r;
        var z = dctZigZag[k];
        zz[z] = receiveAndExtend(s) * (1 << successive);
        k++;
      }
    }

    var successiveACState = 0, successiveACNextValue;

    function decodeACSuccessive(component, zz) {
      var k = spectralStart, e = spectralEnd, r = 0;
      while (k <= e) {
        var z = dctZigZag[k];
        var direction = zz[z] < 0 ? -1 : 1;
        switch (successiveACState) {
        case 0:
          var rs = decodeHuffman(component.huffmanTableAC);
          var s = rs & 15;
          r = rs >> 4;
          if (s === 0) {
            if (r < 15) {
              eobrun = receive(r) + (1 << r);
              successiveACState = 4;
            } else {
              r = 16;
              successiveACState = 1;
            }
          } else {
            if (s !== 1)
              throw new Error("invalid ACn encoding");
            successiveACNextValue = receiveAndExtend(s);
            successiveACState = r ? 2 : 3;
          }
          continue;
        case 1:
        case 2:
          if (zz[z])
            zz[z] += (readBit() << successive) * direction;
          else {
            r--;
            if (r === 0)
              successiveACState = successiveACState == 2 ? 3 : 0;
          }
          break;
        case 3:
          if (zz[z])
            zz[z] += (readBit() << successive) * direction;
          else {
            zz[z] = successiveACNextValue << successive;
            successiveACState = 0;
          }
          break;
        case 4:
          if (zz[z])
            zz[z] += (readBit() << successive) * direction;
          break;
        }
        k++;
      }
      if (successiveACState === 4) {
        eobrun--;
        if (eobrun === 0)
          successiveACState = 0;
      }
    }

    function decodeMcu(component, decode, mcu, row, col) {
      var mcuRow = (mcu / mcusPerLine) | 0;
      var mcuCol = mcu % mcusPerLine;
      var blockRow = mcuRow * component.v + row;
      var blockCol = mcuCol * component.h + col;
      if (component.blocks[blockRow] === undefined && opts.tolerantDecoding)
        return;
      decode(component, component.blocks[blockRow][blockCol]);
    }

    function decodeBlock(component, decode, mcu) {
      var blockRow = (mcu / component.blocksPerLine) | 0;
      var blockCol = mcu % component.blocksPerLine;
      if (component.blocks[blockRow] === undefined && opts.tolerantDecoding)
        return;
      decode(component, component.blocks[blockRow][blockCol]);
    }

    var componentsLength = components.length;
    var component, i, j, k, n;
    var decodeFn;
    if (progressive) {
      if (spectralStart === 0)
        decodeFn = successivePrev === 0 ? decodeDCFirst : decodeDCSuccessive;
      else
        decodeFn = successivePrev === 0 ? decodeACFirst : decodeACSuccessive;
    } else {
      decodeFn = decodeBaseline;
    }

    var mcu = 0, marker;
    var mcuExpected;
    if (componentsLength == 1) {
      mcuExpected = components[0].blocksPerLine * components[0].blocksPerColumn;
    } else {
      mcuExpected = mcusPerLine * frame.mcusPerColumn;
    }
    if (!resetInterval) resetInterval = mcuExpected;

    var h, v;
    while (mcu < mcuExpected) {
      for (i = 0; i < componentsLength; i++)
        components[i].pred = 0;
      eobrun = 0;

      if (componentsLength == 1) {
        component = components[0];
        for (n = 0; n < resetInterval; n++) {
          decodeBlock(component, decodeFn, mcu);
          mcu++;
        }
      } else {
        for (n = 0; n < resetInterval; n++) {
          for (i = 0; i < componentsLength; i++) {
            component = components[i];
            h = component.h;
            v = component.v;
            for (j = 0; j < v; j++) {
              for (k = 0; k < h; k++) {
                decodeMcu(component, decodeFn, mcu, j, k);
              }
            }
          }
          mcu++;
          if (mcu === mcuExpected) break;
        }
      }

      if (mcu === mcuExpected) {
        do {
          if (data[offset] === 0xFF) {
            if (data[offset + 1] !== 0x00) {
              break;
            }
          }
          offset += 1;
        } while (offset < data.length - 2);
      }

      bitsCount = 0;
      marker = (data[offset] << 8) | data[offset + 1];
      if (marker < 0xFF00) {
        throw new Error("marker was not found");
      }

      if (marker >= 0xFFD0 && marker <= 0xFFD7) {
        offset += 2;
      }
      else
        break;
    }

    return offset - startOffset;
  }

  function clampTo8bit(a) {
    return a < 0 ? 0 : a > 255 ? 255 : a;
  }

  function buildComponentData(component) {
    var lines = [];
    var blocksPerLine = component.blocksPerLine;
    var blocksPerColumn = component.blocksPerColumn;
    var samplesPerLine = blocksPerLine << 3;
    var R = new Int32Array(64), r = new Uint8Array(64);

    function quantizeAndInverse(zz, dataOut, dataIn) {
      var qt = component.quantizationTable;
      var v0, v1, v2, v3, v4, v5, v6, v7, t;
      var p = dataIn;
      var i;

      for (i = 0; i < 64; i++)
        p[i] = zz[i] * qt[i];

      for (i = 0; i < 8; ++i) {
        var row = 8 * i;
        if (p[1 + row] == 0 && p[2 + row] == 0 && p[3 + row] == 0 &&
            p[4 + row] == 0 && p[5 + row] == 0 && p[6 + row] == 0 &&
            p[7 + row] == 0) {
          t = (dctSqrt2 * p[0 + row] + 512) >> 10;
          p[0 + row] = t; p[1 + row] = t; p[2 + row] = t; p[3 + row] = t;
          p[4 + row] = t; p[5 + row] = t; p[6 + row] = t; p[7 + row] = t;
          continue;
        }
        v0 = (dctSqrt2 * p[0 + row] + 128) >> 8;
        v1 = (dctSqrt2 * p[4 + row] + 128) >> 8;
        v2 = p[2 + row]; v3 = p[6 + row];
        v4 = (dctSqrt1d2 * (p[1 + row] - p[7 + row]) + 128) >> 8;
        v7 = (dctSqrt1d2 * (p[1 + row] + p[7 + row]) + 128) >> 8;
        v5 = p[3 + row] << 4; v6 = p[5 + row] << 4;
        t = (v0 - v1+ 1) >> 1; v0 = (v0 + v1 + 1) >> 1; v1 = t;
        t = (v2 * dctSin6 + v3 * dctCos6 + 128) >> 8;
        v2 = (v2 * dctCos6 - v3 * dctSin6 + 128) >> 8; v3 = t;
        t = (v4 - v6 + 1) >> 1; v4 = (v4 + v6 + 1) >> 1; v6 = t;
        t = (v7 + v5 + 1) >> 1; v5 = (v7 - v5 + 1) >> 1; v7 = t;
        t = (v0 - v3 + 1) >> 1; v0 = (v0 + v3 + 1) >> 1; v3 = t;
        t = (v1 - v2 + 1) >> 1; v1 = (v1 + v2 + 1) >> 1; v2 = t;
        t = (v4 * dctSin3 + v7 * dctCos3 + 2048) >> 12;
        v4 = (v4 * dctCos3 - v7 * dctSin3 + 2048) >> 12; v7 = t;
        t = (v5 * dctSin1 + v6 * dctCos1 + 2048) >> 12;
        v5 = (v5 * dctCos1 - v6 * dctSin1 + 2048) >> 12; v6 = t;
        p[0 + row] = v0 + v7; p[7 + row] = v0 - v7;
        p[1 + row] = v1 + v6; p[6 + row] = v1 - v6;
        p[2 + row] = v2 + v5; p[5 + row] = v2 - v5;
        p[3 + row] = v3 + v4; p[4 + row] = v3 - v4;
      }

      for (i = 0; i < 8; ++i) {
        var col = i;
        if (p[1*8 + col] == 0 && p[2*8 + col] == 0 && p[3*8 + col] == 0 &&
            p[4*8 + col] == 0 && p[5*8 + col] == 0 && p[6*8 + col] == 0 &&
            p[7*8 + col] == 0) {
          t = (dctSqrt2 * dataIn[i+0] + 8192) >> 14;
          p[0*8 + col] = t; p[1*8 + col] = t; p[2*8 + col] = t; p[3*8 + col] = t;
          p[4*8 + col] = t; p[5*8 + col] = t; p[6*8 + col] = t; p[7*8 + col] = t;
          continue;
        }
        v0 = (dctSqrt2 * p[0*8 + col] + 2048) >> 12;
        v1 = (dctSqrt2 * p[4*8 + col] + 2048) >> 12;
        v2 = p[2*8 + col]; v3 = p[6*8 + col];
        v4 = (dctSqrt1d2 * (p[1*8 + col] - p[7*8 + col]) + 2048) >> 12;
        v7 = (dctSqrt1d2 * (p[1*8 + col] + p[7*8 + col]) + 2048) >> 12;
        v5 = p[3*8 + col]; v6 = p[5*8 + col];
        t = (v0 - v1 + 1) >> 1; v0 = (v0 + v1 + 1) >> 1; v1 = t;
        t = (v2 * dctSin6 + v3 * dctCos6 + 2048) >> 12;
        v2 = (v2 * dctCos6 - v3 * dctSin6 + 2048) >> 12; v3 = t;
        t = (v4 - v6 + 1) >> 1; v4 = (v4 + v6 + 1) >> 1; v6 = t;
        t = (v7 + v5 + 1) >> 1; v5 = (v7 - v5 + 1) >> 1; v7 = t;
        t = (v0 - v3 + 1) >> 1; v0 = (v0 + v3 + 1) >> 1; v3 = t;
        t = (v1 - v2 + 1) >> 1; v1 = (v1 + v2 + 1) >> 1; v2 = t;
        t = (v4 * dctSin3 + v7 * dctCos3 + 2048) >> 12;
        v4 = (v4 * dctCos3 - v7 * dctSin3 + 2048) >> 12; v7 = t;
        t = (v5 * dctSin1 + v6 * dctCos1 + 2048) >> 12;
        v5 = (v5 * dctCos1 - v6 * dctSin1 + 2048) >> 12; v6 = t;
        p[0*8 + col] = v0 + v7; p[7*8 + col] = v0 - v7;
        p[1*8 + col] = v1 + v6; p[6*8 + col] = v1 - v6;
        p[2*8 + col] = v2 + v5; p[5*8 + col] = v2 - v5;
        p[3*8 + col] = v3 + v4; p[4*8 + col] = v3 - v4;
      }

      for (i = 0; i < 64; ++i) {
        var sample = 128 + ((p[i] + 8) >> 4);
        dataOut[i] = sample < 0 ? 0 : sample > 0xFF ? 0xFF : sample;
      }
    }

    requestMemoryAllocation(samplesPerLine * blocksPerColumn * 8);

    var i, j;
    for (var blockRow = 0; blockRow < blocksPerColumn; blockRow++) {
      var scanLine = blockRow << 3;
      for (i = 0; i < 8; i++)
        lines.push(new Uint8Array(samplesPerLine));
      for (var blockCol = 0; blockCol < blocksPerLine; blockCol++) {
        quantizeAndInverse(component.blocks[blockRow][blockCol], r, R);
        var offset = 0, sample = blockCol << 3;
        for (j = 0; j < 8; j++) {
          var line = lines[scanLine + j];
          for (i = 0; i < 8; i++)
            line[sample + i] = r[offset++];
        }
      }
    }
    return lines;
  }

  function JpegImage() {}

  JpegImage.prototype.parse = function(data, opts) {
    this.opts = opts;
    var maxResolutionInPixels = opts.maxResolutionInMP * 1000 * 1000;
    var offset = 0;

    function readUint16() {
      var value = (data[offset] << 8) | data[offset + 1];
      offset += 2;
      return value;
    }

    function readDataBlock() {
      var length = readUint16();
      var array = data.subarray(offset, offset + length - 2);
      offset += array.length;
      return array;
    }

    function prepareComponents(frame) {
      var maxH = 1, maxV = 1;
      var component, componentId;
      for (componentId in frame.components) {
        if (frame.components.hasOwnProperty(componentId)) {
          component = frame.components[componentId];
          if (maxH < component.h) maxH = component.h;
          if (maxV < component.v) maxV = component.v;
        }
      }
      var mcusPerLine = Math.ceil(frame.samplesPerLine / 8 / maxH);
      var mcusPerColumn = Math.ceil(frame.scanLines / 8 / maxV);
      for (componentId in frame.components) {
        if (frame.components.hasOwnProperty(componentId)) {
          component = frame.components[componentId];
          var blocksPerLine = Math.ceil(Math.ceil(frame.samplesPerLine / 8) * component.h / maxH);
          var blocksPerColumn = Math.ceil(Math.ceil(frame.scanLines  / 8) * component.v / maxV);
          var blocksPerLineForMcu = mcusPerLine * component.h;
          var blocksPerColumnForMcu = mcusPerColumn * component.v;
          var blocksToAllocate = blocksPerColumnForMcu * blocksPerLineForMcu;
          var blocks = [];
          requestMemoryAllocation(blocksToAllocate * 256);
          for (var i = 0; i < blocksPerColumnForMcu; i++) {
            var row = [];
            for (var j = 0; j < blocksPerLineForMcu; j++)
              row.push(new Int32Array(64));
            blocks.push(row);
          }
          component.blocksPerLine = blocksPerLine;
          component.blocksPerColumn = blocksPerColumn;
          component.blocks = blocks;
        }
      }
      frame.maxH = maxH;
      frame.maxV = maxV;
      frame.mcusPerLine = mcusPerLine;
      frame.mcusPerColumn = mcusPerColumn;
    }

    var jfif = null;
    var adobe = null;
    var frame, resetInterval;
    var quantizationTables = [], frames = [];
    var huffmanTablesAC = [], huffmanTablesDC = [];
    var fileMarker = readUint16();

    if (fileMarker != 0xFFD8) {
      throw new Error("SOI not found");
    }

    fileMarker = readUint16();
    while (fileMarker != 0xFFD9) {
      var i, j;
      switch(fileMarker) {
        case 0xFF00: break;
        case 0xFFE0: case 0xFFE1: case 0xFFE2: case 0xFFE3:
        case 0xFFE4: case 0xFFE5: case 0xFFE6: case 0xFFE7:
        case 0xFFE8: case 0xFFE9: case 0xFFEA: case 0xFFEB:
        case 0xFFEC: case 0xFFED: case 0xFFEE: case 0xFFEF:
        case 0xFFFE:
          var appData = readDataBlock();
          if (fileMarker === 0xFFE0) {
            if (appData[0] === 0x4A && appData[1] === 0x46 && appData[2] === 0x49 &&
              appData[3] === 0x46 && appData[4] === 0) {
              jfif = {
                version: { major: appData[5], minor: appData[6] },
                densityUnits: appData[7],
                xDensity: (appData[8] << 8) | appData[9],
                yDensity: (appData[10] << 8) | appData[11]
              };
            }
          }
          if (fileMarker === 0xFFEE) {
            if (appData[0] === 0x41 && appData[1] === 0x64 && appData[2] === 0x6F &&
              appData[3] === 0x62 && appData[4] === 0x65 && appData[5] === 0) {
              adobe = {
                version: appData[6],
                flags0: (appData[7] << 8) | appData[8],
                flags1: (appData[9] << 8) | appData[10],
                transformCode: appData[11]
              };
            }
          }
          break;

        case 0xFFDB:
          var quantizationTablesLength = readUint16();
          var quantizationTablesEnd = quantizationTablesLength + offset - 2;
          while (offset < quantizationTablesEnd) {
            var quantizationTableSpec = data[offset++];
            requestMemoryAllocation(64 * 4);
            var tableData = new Int32Array(64);
            if ((quantizationTableSpec >> 4) === 0) {
              for (j = 0; j < 64; j++) {
                var z = dctZigZag[j];
                tableData[z] = data[offset++];
              }
            } else if ((quantizationTableSpec >> 4) === 1) {
              for (j = 0; j < 64; j++) {
                var z = dctZigZag[j];
                tableData[z] = readUint16();
              }
            } else
              throw new Error("DQT: invalid table spec");
            quantizationTables[quantizationTableSpec & 15] = tableData;
          }
          break;

        case 0xFFC0: case 0xFFC1: case 0xFFC2:
          readUint16();
          frame = {};
          frame.extended = (fileMarker === 0xFFC1);
          frame.progressive = (fileMarker === 0xFFC2);
          frame.precision = data[offset++];
          frame.scanLines = readUint16();
          frame.samplesPerLine = readUint16();
          frame.components = {};
          frame.componentsOrder = [];
          var pixelsInFrame = frame.scanLines * frame.samplesPerLine;
          if (pixelsInFrame > maxResolutionInPixels) {
            throw new Error('maxResolutionInMP limit exceeded');
          }
          var componentsCount = data[offset++];
          for (i = 0; i < componentsCount; i++) {
            var componentId = data[offset];
            var h = data[offset + 1] >> 4;
            var v = data[offset + 1] & 15;
            var qId = data[offset + 2];
            if (h <= 0 || v <= 0) {
              throw new Error('Invalid sampling factor');
            }
            frame.componentsOrder.push(componentId);
            frame.components[componentId] = { h: h, v: v, quantizationIdx: qId };
            offset += 3;
          }
          prepareComponents(frame);
          frames.push(frame);
          break;

        case 0xFFC4:
          var huffmanLength = readUint16();
          for (i = 2; i < huffmanLength;) {
            var huffmanTableSpec = data[offset++];
            var codeLengths = new Uint8Array(16);
            var codeLengthSum = 0;
            for (j = 0; j < 16; j++, offset++) {
              codeLengthSum += (codeLengths[j] = data[offset]);
            }
            requestMemoryAllocation(16 + codeLengthSum);
            var huffmanValues = new Uint8Array(codeLengthSum);
            for (j = 0; j < codeLengthSum; j++, offset++)
              huffmanValues[j] = data[offset];
            i += 17 + codeLengthSum;
            ((huffmanTableSpec >> 4) === 0 ?
              huffmanTablesDC : huffmanTablesAC)[huffmanTableSpec & 15] =
              buildHuffmanTable(codeLengths, huffmanValues);
          }
          break;

        case 0xFFDD:
          readUint16();
          resetInterval = readUint16();
          break;

        case 0xFFDC:
          readUint16();
          readUint16();
          break;

        case 0xFFDA:
          readUint16();
          var selectorsCount = data[offset++];
          var components = [], component;
          for (i = 0; i < selectorsCount; i++) {
            component = frame.components[data[offset++]];
            var tableSpec = data[offset++];
            component.huffmanTableDC = huffmanTablesDC[tableSpec >> 4];
            component.huffmanTableAC = huffmanTablesAC[tableSpec & 15];
            components.push(component);
          }
          var spectralStart = data[offset++];
          var spectralEnd = data[offset++];
          var successiveApproximation = data[offset++];
          var processed = decodeScan(data, offset,
            frame, components, resetInterval,
            spectralStart, spectralEnd,
            successiveApproximation >> 4, successiveApproximation & 15, opts);
          offset += processed;
          break;

        case 0xFFFF:
          if (data[offset] !== 0xFF) {
            offset--;
          }
          break;

        default:
          if (data[offset - 3] == 0xFF &&
              data[offset - 2] >= 0xC0 && data[offset - 2] <= 0xFE) {
            offset -= 3;
            break;
          }
          throw new Error("unknown JPEG marker " + fileMarker.toString(16));
      }
      fileMarker = readUint16();
    }

    if (frames.length != 1)
      throw new Error("only single frame JPEGs supported");

    for (i = 0; i < frames.length; i++) {
      var cp = frames[i].components;
      for (var j in cp) {
        cp[j].quantizationTable = quantizationTables[cp[j].quantizationIdx];
        delete cp[j].quantizationIdx;
      }
    }

    this.width = frame.samplesPerLine;
    this.height = frame.scanLines;
    this.jfif = jfif;
    this.adobe = adobe;
    this.components = [];
    for (i = 0; i < frame.componentsOrder.length; i++) {
      var component = frame.components[frame.componentsOrder[i]];
      this.components.push({
        lines: buildComponentData(component),
        scaleX: component.h / frame.maxH,
        scaleY: component.v / frame.maxV
      });
    }
  };

  JpegImage.prototype.getData = function(width, height) {
    var scaleX = this.width / width, scaleY = this.height / height;
    var component1, component2, component3;
    var component1Line, component2Line, component3Line;
    var x, y;
    var offset = 0;
    var Y, Cb, Cr, R, G, B;
    var colorTransform;
    var dataLength = width * height * this.components.length;
    requestMemoryAllocation(dataLength);
    var data = new Uint8Array(dataLength);

    switch (this.components.length) {
      case 1:
        component1 = this.components[0];
        for (y = 0; y < height; y++) {
          component1Line = component1.lines[0 | (y * component1.scaleY * scaleY)];
          for (x = 0; x < width; x++) {
            Y = component1Line[0 | (x * component1.scaleX * scaleX)];
            data[offset++] = Y;
          }
        }
        break;
      case 3:
        colorTransform = true;
        if (this.adobe && this.adobe.transformCode)
          colorTransform = true;
        else if (typeof this.opts.colorTransform !== 'undefined')
          colorTransform = !!this.opts.colorTransform;

        component1 = this.components[0];
        component2 = this.components[1];
        component3 = this.components[2];
        for (y = 0; y < height; y++) {
          component1Line = component1.lines[0 | (y * component1.scaleY * scaleY)];
          component2Line = component2.lines[0 | (y * component2.scaleY * scaleY)];
          component3Line = component3.lines[0 | (y * component3.scaleY * scaleY)];
          for (x = 0; x < width; x++) {
            if (!colorTransform) {
              R = component1Line[0 | (x * component1.scaleX * scaleX)];
              G = component2Line[0 | (x * component2.scaleX * scaleX)];
              B = component3Line[0 | (x * component3.scaleX * scaleX)];
            } else {
              Y = component1Line[0 | (x * component1.scaleX * scaleX)];
              Cb = component2Line[0 | (x * component2.scaleX * scaleX)];
              Cr = component3Line[0 | (x * component3.scaleX * scaleX)];
              R = clampTo8bit(Y + 1.402 * (Cr - 128));
              G = clampTo8bit(Y - 0.3441363 * (Cb - 128) - 0.71413636 * (Cr - 128));
              B = clampTo8bit(Y + 1.772 * (Cb - 128));
            }
            data[offset++] = R;
            data[offset++] = G;
            data[offset++] = B;
          }
        }
        break;
      default:
        throw new Error('Unsupported color mode');
    }
    return data;
  };

  /**
   * Decode JPEG data to RGB pixel array
   * @param {Uint8Array} jpegData - The JPEG binary data
   * @param {Object} opts - Options
   * @returns {Object} {width, height, data: Uint8Array}
   */
  function decode(jpegData, opts) {
    opts = opts || {};
    var defaultOpts = {
      colorTransform: undefined,
      tolerantDecoding: true,
      maxResolutionInMP: 100,
      maxMemoryUsageInMB: 512
    };
    for (var key in defaultOpts) {
      if (opts[key] === undefined) opts[key] = defaultOpts[key];
    }

    var arr = jpegData instanceof Uint8Array ? jpegData : new Uint8Array(jpegData);
    var decoder = new JpegImage();

    totalBytesAllocated = 0;
    maxMemoryUsageBytes = opts.maxMemoryUsageInMB * 1024 * 1024;

    decoder.parse(arr, opts);

    var bytesNeeded = decoder.width * decoder.height * decoder.components.length;
    requestMemoryAllocation(bytesNeeded);

    var rgbData = decoder.getData(decoder.width, decoder.height);

    return {
      width: decoder.width,
      height: decoder.height,
      channels: decoder.components.length,
      data: rgbData
    };
  }

  return {
    decode: decode
  };
})();

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = jpegDecoder;
}
