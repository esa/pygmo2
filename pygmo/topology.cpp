// Copyright 2020 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/s11n.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "object_serialization.hpp"
#include "topology.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

topo_inner<py::object>::topo_inner(const py::object &o)
{
    // Forbid the use of a pygmo.topology as a UDT.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::topology as a UDT is forbidden and prevented by the fact
    // that the generic constructor from UDT is disabled if the input
    // object is a pagmo::topology (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a topology, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input topology.
    if (pygmo::type(o).equal(py::module::import("pygmo").attr("topology"))) {
        pygmo::py_throw(PyExc_TypeError,
                        ("a pygmo.topology cannot be used as a UDT for another pygmo.topology (if you need to copy a "
                         "topology please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "topology");
    check_mandatory_method(o, "get_connections", "topology");
    check_mandatory_method(o, "push_back", "topology");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<topo_inner_base> topo_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<topo_inner>(m_value);
}

std::pair<std::vector<std::size_t>, vector_double> topo_inner<py::object>::get_connections(std::size_t n) const
{
    // NOTE: get_connections() may be called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    auto topo_name = get_name();

    // Fetch the connections in Python form.
    auto o = py::cast<py::iterable>(m_value.attr("get_connections")(n));

    // Prepare the return value.
    std::pair<std::vector<std::size_t>, vector_double> retval;

    // We will try to interpret o as a collection of generic python objects.
    auto begin = std::begin(o);
    const auto end = std::end(o);

    if (begin == end) {
        // Empty iteratable.
        pygmo::py_throw(PyExc_ValueError, ("the iteratable returned by a topology of type '" + topo_name
                                           + "' is empty (it should contain 2 elements)")
                                              .c_str());
    }

    retval.first = pygmo::ndarr_to_vector<std::vector<std::size_t>>(py::cast<py::array_t<std::size_t>>(*begin));

    if (++begin == end) {
        // Only one element in the iteratable.
        pygmo::py_throw(PyExc_ValueError, ("the iteratable returned by a topology of type '" + topo_name
                                           + "' has only 1 element (it should contain 2 elements)")
                                              .c_str());
    }

    retval.second = pygmo::ndarr_to_vector<vector_double>(py::cast<py::array_t<double>>(*begin));

    if (++begin != end) {
        // Too many elements.
        pygmo::py_throw(PyExc_ValueError, ("the iteratable returned by a topology of type '" + topo_name
                                           + "' has more than 2 elements (it should contain 2 elements)")
                                              .c_str());
    }

    return retval;
}

void topo_inner<py::object>::push_back()
{
    m_value.attr("push_back")();
}

std::string topo_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string topo_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

template <typename Archive>
void topo_inner<py::object>::save(Archive &ar, unsigned) const
{
    ar << boost::serialization::base_object<topo_inner_base>(*this);
    ar << pygmo::object_to_vchar(m_value);
}

template <typename Archive>
void topo_inner<py::object>::load(Archive &ar, unsigned)
{
    ar >> boost::serialization::base_object<topo_inner_base>(*this);
    std::vector<char> v;
    ar >> v;
    m_value = pygmo::vchar_to_object(v);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_IMPLEMENT(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

// Serialization support for the topology class.
py::tuple topology_pickle_getstate(const pagmo::topology &t)
{
    // The idea here is that first we extract a char array
    // into which t has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << t;
    }
    auto s = oss.str();
    return py::make_tuple(py::bytes(s.data(), boost::numeric_cast<py::size_t>(s.size())));
}

pagmo::topology topology_pickle_setstate(py::tuple state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialized the object.
    if (py::len(state) != 1) {
        pygmo::py_throw(PyExc_ValueError, ("the state tuple passed for topology deserialization "
                                           "must have 1 element, but instead it has "
                                           + std::to_string(py::len(state)) + " element(s)")
                                              .c_str());
    }

    auto ptr = PyBytes_AsString(state[0].ptr());
    if (!ptr) {
        pygmo::py_throw(PyExc_TypeError, "a bytes object is needed to deserialize a topology");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len(state[0])));
    pagmo::topology t;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> t;
    }

    return t;
}

} // namespace pygmo
