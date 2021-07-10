#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 <+YOU OR YOUR COMPANY+>.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
from gnuradio import uhd
import ieee802_11
from gnuradio import fft
from gnuradio.fft import window
#import wifi_test_swig as wifi_test
import time
import optparse
import foo
save_time=time.strftime("%m-%d-%H:%M", time.localtime())

parser=optparse.OptionParser("'usage -c <channel frequency> ")
parser.add_option('-c',dest='channel',type='string',help='set channel frequency')
(options,args)=parser.parse_args()
if (options.channel==None):
    print(parser.usage)
    exit(0)
Freq=int(options.channel)

class qa_fre_offset (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_t (self):
        # set up fg
        samp_rate=20e6
        #5745 149 2412 1  2447 8  5805 161 6 2437 11 2462 2432for iot_dev
        freq = Freq
        lo_offset = 0
        gain=0.75
        sync_length = 320
        window_size = 48
        chan_est= 0
        uhd_usrp_source_0 = uhd.usrp_source(
        	"addr=192.168.10.2",
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        uhd_usrp_source_0.set_samp_rate(samp_rate)
        uhd_usrp_source_0.set_center_freq(uhd.tune_request(freq, rf_freq = freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        uhd_usrp_source_0.set_normalized_gain(gain, 0)
        #detect frame
        blocks_divide_xx_0 = blocks.divide_ff(1)
        blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, 16)
        blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)
        blocks_conjugate_cc_0 = blocks.conjugate_cc()
        blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        ieee802_11_moving_average_xx_1 = ieee802_11.moving_average_ff(window_size + 16)
        ieee802_11_moving_average_xx_0 = ieee802_11.moving_average_cc(window_size)
        blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        sync_short=ieee802_11.sync_short(0.56, 2, False, False)
        sync_long = ieee802_11.sync_long(sync_length, False, False)
        stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        fft_vxx_0= fft.fft_vcc(64, True, (window.rectangular(64)), True, 1)
        frame_equalizer = ieee802_11.frame_equalizer(chan_est, freq, samp_rate, False, False)
        decode_mac= ieee802_11.decode_mac(False, False)
        parse_mac= ieee802_11.parse_mac(False, False)
        blocks_pdu_to_tagged_stream_1 = blocks.pdu_to_tagged_stream(blocks.complex_t, 'packet_len')

        #sink
        tag_equalizer=blocks.tag_debug(48,"equalizexxxxx")

        decode_mac_msg_extract=blocks.message_debug()

        after_sync_long=blocks.file_sink(8,"after_sync_long"+save_time+".bin")
        after_sync_long_fft=blocks.file_sink(64*8,"after_sync_long_fft"+save_time+".bin")
        after_sync_short=blocks.file_sink(8,"after_sync_short"+save_time+".bin")
        after_equalizer_bytes=blocks.file_sink(48,"after_equalizer_bytes"+save_time+".bin")
        after_equalizer_symbols=blocks.file_sink(8,"after_equalizer_symbols"+save_time+".bin")

        #raw_sink=blocks.file_sink(8,"after_usrp_370_osx.bin")
        tag_sync_long=blocks.tag_debug(8,"synclongxxx")

        #gnerate pcap
        foo_wireshark_connector_0 = foo.wireshark_connector(127, False)
        blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*1, 'wifi_'+save_time+'.pcap', True)

        #vector_sink=blocks.vector_sink_c()
        #connect
        self.tb.connect((uhd_usrp_source_0, 0), (blocks_multiply_xx_0, 0))
        self.tb.connect(uhd_usrp_source_0, blocks_complex_to_mag_squared_0)
        self.tb.connect(uhd_usrp_source_0, blocks_delay_0_0)
        self.tb.connect((blocks_complex_to_mag_0, 0), (blocks_divide_xx_0, 0))
        self.tb.connect((blocks_complex_to_mag_squared_0, 0), (ieee802_11_moving_average_xx_1, 0))
        self.tb.connect((blocks_conjugate_cc_0, 0), (blocks_multiply_xx_0, 1))
        self.tb.connect((blocks_delay_0_0, 0), (blocks_conjugate_cc_0, 0))
        self.tb.connect((blocks_multiply_xx_0, 0), (ieee802_11_moving_average_xx_0, 0))
        self.tb.connect((ieee802_11_moving_average_xx_0, 0), (blocks_complex_to_mag_0, 0))
        self.tb.connect((ieee802_11_moving_average_xx_0, 0), (sync_short, 1))
        self.tb.connect((ieee802_11_moving_average_xx_1, 0), (blocks_divide_xx_0, 1))
        self.tb.connect((blocks_delay_0_0, 0), (sync_short, 0))
        self.tb.connect((blocks_divide_xx_0, 0), (sync_short, 2))
        self.tb.connect((sync_short, 0), (blocks_delay_0, 0))
        self.tb.connect((blocks_delay_0, 0), (sync_long, 1))
        self.tb.connect((sync_short, 0), (sync_long, 0))
        self.tb.connect((sync_long, 0), (stream_to_vector, 0))
        self.tb.connect((stream_to_vector, 0), (fft_vxx_0, 0))
        self.tb.connect((fft_vxx_0, 0), (frame_equalizer, 0))
        self.tb.connect((frame_equalizer, 0), (decode_mac, 0))

        self.tb.msg_connect((decode_mac, 'out'), (parse_mac, 'in'))

        self.tb.msg_connect((frame_equalizer, 'symbols'), (blocks_pdu_to_tagged_stream_1, 'pdus'))

        self.tb.msg_connect((decode_mac, 'out'), (foo_wireshark_connector_0, 'in'))
        self.tb.connect((foo_wireshark_connector_0, 0), (blocks_file_sink_0, 0))

        #debug

        #self.tb.msg_connect((decode_mac, 'out'), (decode_mac_msg_extract, 'print'))
        self.tb.msg_connect((parse_mac, 'fer'), (decode_mac_msg_extract, 'print'))
        #self.tb.connect(sync_long, tag_equalizer)
        #self.tb.connect(frame_equalizer, after_equalizer_bytes)
        #这里出来的才是复数symbols
        self.tb.connect((blocks_pdu_to_tagged_stream_1, 0), (after_equalizer_symbols, 0))
        self.tb.connect(sync_short,after_sync_short)
        #self.tb.connect(sync_long,after_sync_long)
        #self.tb.connect(fft_vxx_0,after_sync_long_fft)
        #self.tb.connect(uhd_usrp_source_0 , raw_sink)
        #self.tb.connect(frame_equalizer, tag_equalizer)

        
        #self.tb.connect(sync_long, tag_sync_long)
        #self.tb.connect(fre_offset, vector_sink)
        self.tb.run ()
        # check data


if __name__ == '__main__':
    gr_unittest.run(qa_fre_offset, "qa_fre_offset.xml")
