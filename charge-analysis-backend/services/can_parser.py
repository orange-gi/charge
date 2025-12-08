"""CAN 日志解析器 - 基于 DBC 文件解析 BLF 日志"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cantools
import can
import pandas as pd

# 尝试导入不同格式的解析库
try:
    from asammdf import MDF
    HAS_ASAMMDF = True
except ImportError:
    HAS_ASAMMDF = False

logger = logging.getLogger(__name__)


class CANLogParser:
    """CAN 日志解析器，使用 DBC 文件解析 BLF 日志"""
    
    def __init__(self, dbc_file_path: Optional[str] = None):
        """初始化解析器
        
        Args:
            dbc_file_path: DBC 文件路径，如果为 None，则使用默认的 GBT_27930_2015.dbc
        """
        if dbc_file_path is None:
            # 使用默认的 DBC 文件
            # 尝试多个可能的路径
            possible_paths = [
                Path(__file__).parent / "GBT_27930_2015.dbc",  # 相对于当前文件
                Path("services/GBT_27930_2015.dbc"),  # 相对于工作目录
                Path("../services/GBT_27930_2015.dbc"),  # 上级目录
            ]
            
            for path in possible_paths:
                if path.exists():
                    dbc_file_path = str(path)
                    break
            
            if dbc_file_path is None:
                # 如果都找不到，使用第一个路径（会在 _load_dbc 中抛出错误）
                dbc_file_path = str(possible_paths[0])
        
        self.dbc_file_path = Path(dbc_file_path)
        self.db = None
        # 性能优化：预先构建信号到消息ID的映射
        self._signal_to_message_ids: Dict[str, List[int]] = {}
        self._message_id_to_message: Dict[int, Any] = {}
        self._load_dbc()
    
    def _load_dbc(self) -> None:
        """加载 DBC 文件并构建性能优化映射"""
        try:
            if not self.dbc_file_path.exists():
                raise FileNotFoundError(f"DBC 文件不存在: {self.dbc_file_path}")
            
            logger.info(f"加载 DBC 文件: {self.dbc_file_path}")
            self.db = cantools.database.load_file(str(self.dbc_file_path))
            logger.info(f"DBC 文件加载成功，包含 {len(self.db.messages)} 个消息定义")
            
            # 性能优化：预先构建映射
            self._build_performance_mappings()
            
        except Exception as e:
            logger.error(f"加载 DBC 文件失败: {e}")
            raise
    
    def _build_performance_mappings(self) -> None:
        """构建信号到消息ID的映射，用于快速查找"""
        self._signal_to_message_ids = {}
        self._message_id_to_message = {}
        
        for message in self.db.messages:
            can_id = message.frame_id
            self._message_id_to_message[can_id] = message
            
            for signal in message.signals:
                signal_name = signal.name
                if signal_name not in self._signal_to_message_ids:
                    self._signal_to_message_ids[signal_name] = []
                self._signal_to_message_ids[signal_name].append(can_id)
        
        logger.info(f"构建映射完成：{len(self._message_id_to_message)} 个消息，{len(self._signal_to_message_ids)} 个信号")
    
    def get_message_ids_for_signals(self, signal_names: List[str]) -> set[int]:
        """获取包含指定信号的 CAN ID 集合"""
        message_ids = set()
        for signal_name in signal_names:
            if signal_name in self._signal_to_message_ids:
                message_ids.update(self._signal_to_message_ids[signal_name])
        return message_ids
    
    async def parse_blf(
        self,
        blf_file_path: str,
        filter_signals: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """解析 BLF 日志文件
        
        支持 Vector CANoe BLF 格式和 ASAM MDF 格式
        
        Args:
            blf_file_path: BLF 文件路径
            filter_signals: 要提取的信号名称列表，如果为 None 则提取所有信号
            progress_callback: 进度回调函数 (stage, progress, message)
        
        Returns:
            DataFrame，包含解析后的信号数据
        """
        blf_path = Path(blf_file_path)
        
        if not blf_path.exists():
            raise FileNotFoundError(f"BLF 文件不存在: {blf_path}")
        
        if progress_callback:
            await progress_callback("CAN解析", 10, "检测 BLF 文件格式...")
        
        # 检测文件格式
        file_format = await self._detect_blf_format(blf_path)
        logger.info(f"检测到 BLF 文件格式: {file_format}")
        
        if file_format == "vector":
            # Vector CANoe BLF 格式
            return await self._parse_vector_blf(blf_path, filter_signals, progress_callback)
        elif file_format == "asam" and HAS_ASAMMDF:
            # ASAM MDF 格式
            return await self._parse_asam_mdf(blf_path, filter_signals, progress_callback)
        else:
            raise ValueError(f"不支持的 BLF 文件格式: {file_format}")
    
    async def _detect_blf_format(self, file_path: Path) -> str:
        """检测 BLF 文件格式"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            # Vector BLF 格式: 以 "LOGG" 开头
            if header.startswith(b'LOGG'):
                return "vector"
            # ASAM MDF 格式: 以特定标识符开头
            elif header.startswith(b'MDF ') or HAS_ASAMMDF:
                try:
                    # 尝试作为 MDF 文件打开
                    from asammdf import MDF
                    with MDF(file_path):
                        return "asam"
                except:
                    return "vector"
            else:
                return "unknown"
        except Exception as e:
            logger.warning(f"检测文件格式失败: {e}，默认使用 Vector 格式")
            return "vector"
    
    async def _parse_vector_blf(
        self,
        blf_path: Path,
        filter_signals: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """解析 Vector CANoe BLF 格式文件（高性能版本）"""
        logger.info(f"开始解析 Vector BLF 文件: {blf_path}")
        logger.info(f"文件大小: {blf_path.stat().st_size / (1024*1024):.2f} MB")
        
        if progress_callback:
            await progress_callback("CAN解析", 10, "准备解析 Vector BLF 文件...")
        
        try:
            # 性能优化：如果指定了信号，只关注包含这些信号的消息ID
            target_message_ids = None
            if filter_signals:
                logger.info(f"信号过滤模式: 仅解析 {len(filter_signals)} 个指定信号")
                logger.debug(f"目标信号列表: {filter_signals[:10]}{'...' if len(filter_signals) > 10 else ''}")
                target_message_ids = self.get_message_ids_for_signals(filter_signals)
                logger.info(f"找到 {len(target_message_ids)} 个包含目标信号的 CAN ID")
                if target_message_ids:
                    logger.debug(f"目标 CAN ID 示例: {list(target_message_ids)[:5]}")
                if not target_message_ids:
                    logger.warning(f"未找到包含指定信号的消息，将解析所有消息")
            else:
                logger.info("全量解析模式: 解析所有信号")
            
            if progress_callback:
                await progress_callback("CAN解析", 15, "读取 Vector BLF 文件...")
            
            # 性能优化：使用流式读取，不一次性加载所有消息到内存
            log = can.BLFReader(str(blf_path))
            
            # 先快速统计消息数量（用于进度显示）
            message_count = 0
            if progress_callback:
                await progress_callback("CAN解析", 20, "统计消息数量...")
                logger.info("开始统计 BLF 文件中的消息数量...")
                start_time = asyncio.get_event_loop().time()
                message_count = sum(1 for _ in log)
                count_time = asyncio.get_event_loop().time() - start_time
                log = can.BLFReader(str(blf_path))  # 重新打开文件
                logger.info(f"统计完成: BLF 文件包含 {message_count:,} 条消息 (耗时 {count_time:.2f}秒)")
                if message_count > 0:
                    logger.info(f"平均消息速率: {message_count/count_time:.0f} 消息/秒")
            
            if progress_callback:
                await progress_callback("CAN解析", 25, f"开始解码消息...")
            
            parsed_data = []
            processed_count = 0
            decoded_count = 0
            skipped_count = 0
            error_count = 0
            last_progress_time = asyncio.get_event_loop().time()
            parse_start_time = asyncio.get_event_loop().time()
            
            logger.info("开始流式处理 CAN 消息...")
            
            # 流式处理消息
            for msg in log:
                try:
                    can_id = msg.arbitration_id
                    
                    # 性能优化：如果指定了目标消息ID，快速跳过不相关的消息
                    if target_message_ids is not None and can_id not in target_message_ids:
                        skipped_count += 1
                        processed_count += 1
                        continue
                    
                    # 性能优化：使用预构建的映射直接获取消息
                    message = self._message_id_to_message.get(can_id)
                    if message is None:
                        skipped_count += 1
                        processed_count += 1
                        continue
                    
                    # 解码消息
                    try:
                        decoded = message.decode(msg.data, decode_choices=False)
                        decoded_count += 1
                        
                        # 详细日志：每10000条记录一次详细信息
                        if decoded_count % 10000 == 0:
                            logger.debug(f"已解码 {decoded_count:,} 条消息，当前: {message.name} (ID: 0x{can_id:03X})")
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"解码消息失败: {message.name} (ID: 0x{can_id:03X}), 错误: {e}")
                        processed_count += 1
                        continue
                    
                    # 构建数据记录
                    record = {
                        'timestamp': pd.Timestamp.fromtimestamp(msg.timestamp),
                        'can_id': hex(can_id),
                        'message_name': message.name,
                        'dlc': msg.dlc,
                    }
                    
                    # 添加信号值（只添加过滤的信号）
                    if filter_signals:
                        # 只添加用户选择的信号
                        has_signal = False
                        for signal_name, signal_value in decoded.items():
                            if signal_name in filter_signals:
                                record[signal_name] = signal_value
                                has_signal = True
                        if has_signal:
                            parsed_data.append(record)
                    else:
                        # 添加所有信号
                        for signal_name, signal_value in decoded.items():
                            record[signal_name] = signal_value
                        parsed_data.append(record)
                    
                    processed_count += 1
                    
                    # 性能优化：减少进度更新频率（每5000条或每0.5秒更新一次）
                    if progress_callback:
                        current_time = asyncio.get_event_loop().time()
                        if processed_count % 5000 == 0 or (current_time - last_progress_time) >= 0.5:
                            if message_count > 0:
                                progress = 25 + int(70 * (processed_count / message_count))
                            else:
                                progress = min(90, 25 + int(processed_count / 100))
                            
                            # 计算处理速度
                            elapsed = current_time - parse_start_time
                            speed = processed_count / elapsed if elapsed > 0 else 0
                            
                            await progress_callback(
                                "CAN解析",
                                progress,
                                f"已处理 {processed_count:,}/{message_count:,} 条消息，解码 {decoded_count:,} 条 ({speed:.0f} 消息/秒)..."
                            )
                            last_progress_time = current_time
                            
                            # 每50000条记录一次详细统计
                            if processed_count % 50000 == 0:
                                logger.info(
                                    f"解析进度: {processed_count:,}/{message_count:,} ({processed_count*100/message_count:.1f}%), "
                                    f"解码: {decoded_count:,}, 跳过: {skipped_count:,}, 错误: {error_count}, "
                                    f"速度: {speed:.0f} 消息/秒"
                                )
                
                except Exception as e:
                    logger.debug(f"处理消息失败: {e}")
                    processed_count += 1
                    continue
            
            if progress_callback:
                await progress_callback("CAN解析", 95, "正在构建数据框...")
            
            parse_time = asyncio.get_event_loop().time() - parse_start_time
            
            # 输出详细统计信息
            logger.info("=" * 60)
            logger.info("CAN 消息解析统计:")
            logger.info(f"  总消息数: {processed_count:,}")
            logger.info(f"  成功解码: {decoded_count:,} ({decoded_count*100/processed_count:.1f}%)" if processed_count > 0 else "  成功解码: 0")
            logger.info(f"  跳过消息: {skipped_count:,} ({skipped_count*100/processed_count:.1f}%)" if processed_count > 0 else "  跳过消息: 0")
            logger.info(f"  解码错误: {error_count} ({error_count*100/processed_count:.1f}%)" if processed_count > 0 else "  解码错误: 0")
            logger.info(f"  解析耗时: {parse_time:.2f} 秒")
            if parse_time > 0:
                logger.info(f"  处理速度: {processed_count/parse_time:.0f} 消息/秒")
            logger.info("=" * 60)
            
            if not parsed_data:
                logger.error("未能从 BLF 文件中解析出任何 CAN 消息数据")
                raise ValueError("未能从 BLF 文件中解析出任何 CAN 消息数据")
            
            logger.info("开始构建 pandas DataFrame...")
            df_start_time = asyncio.get_event_loop().time()
            df = pd.DataFrame(parsed_data)
            df_time = asyncio.get_event_loop().time() - df_start_time
            
            # 确保时间戳列
            if 'timestamp' in df.columns:
                df['ts'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"DataFrame 构建完成: {len(df):,} 行 × {len(df.columns)} 列 (耗时 {df_time:.2f}秒)")
            logger.info(f"数据列: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
            logger.info(f"内存占用: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            if progress_callback:
                await progress_callback("CAN解析", 100, f"CAN 解析完成: {len(df):,} 条记录，{len(df.columns)} 个信号")
            
            total_time = asyncio.get_event_loop().time() - parse_start_time
            logger.info(f"解析完成，总耗时: {total_time:.2f} 秒，共 {len(df):,} 条记录，{len(df.columns)} 个信号")
            return df
            
        except Exception as e:
            logger.error(f"解析 Vector BLF 文件失败: {e}", exc_info=True)
            raise
    
    async def _parse_asam_mdf(
        self,
        mdf_path: Path,
        filter_signals: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """解析 ASAM MDF 格式文件"""
        if not HAS_ASAMMDF:
            raise ImportError("asammdf 库未安装，无法解析 ASAM MDF 格式")
        
        if progress_callback:
            await progress_callback("CAN解析", 15, "读取 ASAM MDF 文件...")
        
        try:
            mdf = MDF(mdf_path)
            logger.info(f"MDF 文件读取成功: {mdf_path}")
            
            if progress_callback:
                await progress_callback("CAN解析", 20, "开始解析 CAN 消息...")
            
            parsed_data = []
            total_channels = len(mdf.channels_db)
            
            for idx, channel_name in enumerate(mdf.channels_db):
                try:
                    signals = mdf.get(channel_name)
                    
                    message_data = await self._parse_can_message(
                        channel_name,
                        signals,
                        filter_signals
                    )
                    
                    if message_data:
                        parsed_data.extend(message_data)
                    
                    if progress_callback and idx % 100 == 0:
                        progress = 20 + int(70 * (idx / total_channels))
                        await progress_callback(
                            "CAN解析",
                            progress,
                            f"已处理 {idx}/{total_channels} 个通道..."
                        )
                
                except Exception as e:
                    logger.warning(f"解析通道 {channel_name} 失败: {e}")
                    continue
            
            if progress_callback:
                await progress_callback("CAN解析", 90, "正在构建数据框...")
            
            if not parsed_data:
                logger.warning("未解析到任何数据，使用备用方法")
                return await self._parse_blf_fallback(mdf_path, filter_signals)
            
            df = pd.DataFrame(parsed_data)
            
            if 'timestamp' in df.columns:
                df['ts'] = pd.to_datetime(df['timestamp'])
            elif 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'])
            
            if progress_callback:
                await progress_callback("CAN解析", 100, "CAN 解析完成")
            
            logger.info(f"解析完成，共 {len(df)} 条记录，{len(df.columns)} 个信号")
            return df
            
        except Exception as e:
            logger.error(f"解析 ASAM MDF 文件失败: {e}", exc_info=True)
            raise
    
    async def _parse_can_message(
        self,
        channel_name: str,
        signals: Any,
        filter_signals: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """解析单个 CAN 消息
        
        Args:
            channel_name: 通道名称
            signals: 信号数据
            filter_signals: 要提取的信号名称列表
        
        Returns:
            解析后的消息数据列表
        """
        parsed_messages = []
        
        try:
            # 尝试从通道名称提取 CAN ID
            can_id = self._extract_can_id_from_channel(channel_name)
            
            if can_id is None:
                return parsed_messages
            
            # 在 DBC 数据库中查找匹配的消息
            message = None
            for msg in self.db.messages:
                if msg.frame_id == can_id or hex(msg.frame_id) == can_id:
                    message = msg
                    break
            
            if message is None:
                return parsed_messages
            
            # 解析信号
            if hasattr(signals, 'timestamps') and hasattr(signals, 'samples'):
                timestamps = signals.timestamps
                samples = signals.samples
                
                # 如果 samples 是数组，需要逐个解析
                if len(samples) > 0:
                    for i, sample in enumerate(samples):
                        try:
                            # 将 CAN 数据转换为消息
                            if isinstance(sample, (bytes, bytearray)):
                                decoded = message.decode(sample)
                            elif isinstance(sample, int):
                                decoded = message.decode(sample.to_bytes(8, 'little'))
                            else:
                                continue
                            
                            # 构建数据记录
                            record = {
                                'timestamp': timestamps[i] if i < len(timestamps) else None,
                                'can_id': can_id,
                                'message_name': message.name,
                            }
                            
                            # 添加信号值
                            for signal_name, signal_value in decoded.items():
                                if filter_signals is None or signal_name in filter_signals:
                                    record[signal_name] = signal_value
                            
                            parsed_messages.append(record)
                            
                        except Exception as e:
                            logger.debug(f"解码消息失败: {e}")
                            continue
            
        except Exception as e:
            logger.debug(f"解析 CAN 消息失败 {channel_name}: {e}")
        
        return parsed_messages
    
    def _extract_can_id_from_channel(self, channel_name: str) -> Optional[int]:
        """从通道名称提取 CAN ID"""
        try:
            # 常见的通道命名格式：CAN1_0x123 或 0x123 或 123
            import re
            
            # 尝试提取十六进制 ID
            hex_match = re.search(r'0x([0-9a-fA-F]+)', channel_name, re.IGNORECASE)
            if hex_match:
                return int(hex_match.group(1), 16)
            
            # 尝试提取十进制 ID
            dec_match = re.search(r'\b(\d+)\b', channel_name)
            if dec_match:
                return int(dec_match.group(1))
            
        except Exception:
            pass
        
        return None
    
    async def _parse_blf_fallback(
        self,
        blf_path: Path,
        filter_signals: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """备用解析方法：直接处理文件
        
        当标准解析方法失败时使用此方法
        """
        logger.info("使用备用方法解析文件")
        
        # 尝试作为 Vector BLF 文件解析
        try:
            log = can.BLFReader(str(blf_path))
            messages = list(log)
            
            parsed_data = []
            for msg in messages:
                try:
                    message = None
                    for db_msg in self.db.messages:
                        if db_msg.frame_id == msg.arbitration_id:
                            message = db_msg
                            break
                    
                    if message:
                        decoded = message.decode(msg.data, decode_choices=False)
                        record = {
                            'timestamp': pd.Timestamp.fromtimestamp(msg.timestamp),
                            'can_id': hex(msg.arbitration_id),
                            'message_name': message.name,
                        }
                        for signal_name, signal_value in decoded.items():
                            if filter_signals is None or signal_name in filter_signals:
                                record[signal_name] = signal_value
                        parsed_data.append(record)
                except:
                    continue
            
            if parsed_data:
                df = pd.DataFrame(parsed_data)
                df['ts'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            logger.debug(f"备用 Vector BLF 解析失败: {e}")
        
        # 如果 Vector BLF 失败，尝试作为 ASAM MDF
        if HAS_ASAMMDF:
            try:
                mdf = MDF(blf_path)
                
                # 获取所有通道
                all_channels = list(mdf.channels_db.keys())
                logger.info(f"找到 {len(all_channels)} 个通道")
                
                # 提取信号数据
                signal_data = {}
                timestamps = None
                
                for channel_name in all_channels:
                    try:
                        channel = mdf.get(channel_name)
                        
                        if hasattr(channel, 'timestamps'):
                            if timestamps is None:
                                timestamps = channel.timestamps
                            
                            # 检查是否是我们需要的信号
                            if filter_signals is None or any(
                                sig in channel_name for sig in filter_signals
                            ):
                                signal_data[channel_name] = channel.samples
                        
                    except Exception as e:
                        logger.debug(f"提取通道 {channel_name} 失败: {e}")
                        continue
                
                # 构建 DataFrame
                if signal_data:
                    df = pd.DataFrame(signal_data)
                    if timestamps is not None and len(timestamps) == len(df):
                        df['timestamp'] = pd.to_datetime(timestamps, unit='s')
                        df['ts'] = df['timestamp']
                    return df
            except Exception as e:
                logger.debug(f"备用 ASAM MDF 解析失败: {e}")
        
        # 如果所有方法都失败，返回空 DataFrame
        logger.warning("所有解析方法都失败，返回空数据框")
        return pd.DataFrame()
    
    def get_available_signals(self) -> List[str]:
        """获取 DBC 文件中所有可用的信号名称"""
        signals = []
        for message in self.db.messages:
            for signal in message.signals:
                signals.append(signal.name)
        return sorted(set(signals))
    
    def get_available_messages(self) -> List[Dict[str, Any]]:
        """获取 DBC 文件中所有可用的消息信息"""
        messages = []
        for message in self.db.messages:
            messages.append({
                'name': message.name,
                'frame_id': message.frame_id,
                'length': message.length,
                'signals': [sig.name for sig in message.signals]
            })
        return messages

