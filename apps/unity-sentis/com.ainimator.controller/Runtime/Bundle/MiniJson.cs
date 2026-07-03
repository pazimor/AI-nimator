using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Minimal, dependency-free JSON reader used to parse
    /// <c>norm_stats.json</c>, whose nested float arrays (e.g.
    /// <c>bone_mean: [[[[...]]]]</c>) are not representable by Unity's
    /// <c>JsonUtility</c> (it does not support nested/jagged arrays).
    /// </summary>
    /// <remarks>
    /// This parser only supports the subset of JSON needed by the bundle
    /// files: objects, arrays, numbers, strings, booleans and null. It is
    /// intentionally small and allocation-light rather than general-purpose.
    /// </remarks>
    internal static class MiniJson
    {
        /// <summary>Parse a JSON document into a tree of plain CLR objects.</summary>
        /// <param name="json">Raw JSON text.</param>
        /// <returns>
        /// <see cref="Dictionary{TKey,TValue}"/> for objects,
        /// <see cref="List{T}"/> for arrays, <see cref="double"/> for numbers,
        /// <see cref="string"/> for strings, <see cref="bool"/> for booleans,
        /// <c>null</c> for JSON null.
        /// </returns>
        public static object Parse(string json)
        {
            var index = 0;
            var value = ParseValue(json, ref index);
            return value;
        }

        private static object ParseValue(string s, ref int i)
        {
            SkipWhitespace(s, ref i);
            var c = s[i];
            switch (c)
            {
                case '{': return ParseObject(s, ref i);
                case '[': return ParseArray(s, ref i);
                case '"': return ParseString(s, ref i);
                case 't':
                    i += 4;
                    return true;
                case 'f':
                    i += 5;
                    return false;
                case 'n':
                    i += 4;
                    return null;
                default: return ParseNumber(s, ref i);
            }
        }

        private static Dictionary<string, object> ParseObject(string s, ref int i)
        {
            var result = new Dictionary<string, object>();
            i++; // '{'
            SkipWhitespace(s, ref i);
            if (s[i] == '}')
            {
                i++;
                return result;
            }

            while (true)
            {
                SkipWhitespace(s, ref i);
                var key = ParseString(s, ref i);
                SkipWhitespace(s, ref i);
                i++; // ':'
                var value = ParseValue(s, ref i);
                result[key] = value;
                SkipWhitespace(s, ref i);
                if (s[i] == ',')
                {
                    i++;
                    continue;
                }

                i++; // '}'
                break;
            }

            return result;
        }

        private static List<object> ParseArray(string s, ref int i)
        {
            var result = new List<object>();
            i++; // '['
            SkipWhitespace(s, ref i);
            if (s[i] == ']')
            {
                i++;
                return result;
            }

            while (true)
            {
                var value = ParseValue(s, ref i);
                result.Add(value);
                SkipWhitespace(s, ref i);
                if (s[i] == ',')
                {
                    i++;
                    continue;
                }

                i++; // ']'
                break;
            }

            return result;
        }

        private static string ParseString(string s, ref int i)
        {
            var sb = new StringBuilder();
            i++; // opening quote
            while (s[i] != '"')
            {
                if (s[i] == '\\')
                {
                    i++;
                    switch (s[i])
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case '"': sb.Append('"'); break;
                        case '\\': sb.Append('\\'); break;
                        case '/': sb.Append('/'); break;
                        case 'u':
                            var hex = s.Substring(i + 1, 4);
                            sb.Append((char)ushort.Parse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                            i += 4;
                            break;
                        default: sb.Append(s[i]); break;
                    }

                    i++;
                }
                else
                {
                    sb.Append(s[i]);
                    i++;
                }
            }

            i++; // closing quote
            return sb.ToString();
        }

        private static double ParseNumber(string s, ref int i)
        {
            var start = i;
            while (i < s.Length && (char.IsDigit(s[i]) || s[i] is '-' or '+' or '.' or 'e' or 'E'))
            {
                i++;
            }

            return double.Parse(s.Substring(start, i - start), CultureInfo.InvariantCulture);
        }

        private static void SkipWhitespace(string s, ref int i)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i]))
            {
                i++;
            }
        }
    }
}
